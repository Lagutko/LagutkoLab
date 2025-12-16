import subprocess as sp
import numpy as np
import cv2
import threading
import queue
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from datetime import datetime
import os
import torch
from sqlalchemy import create_engine, Column, Integer, String, TIMESTAMP, func, text
from sqlalchemy.orm import sessionmaker, declarative_base
import mediapipe as mp
import re
import logging

# ----------- НАСТРОЙКИ -----------
RTSP_URL = "rtsp://admin:parol1@10.00.00.00"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
MAX_QUEUE_SIZE = 10
ALERT_TIME = 6 * 60
model_path = r'C:/Project/runs/detect/camera1 dop new model/weights/best.pt'
screenshot_dir = "C:/Qpax_View/qpax_view/media"
os.makedirs(screenshot_dir, exist_ok=True)

# ----------- БАЗА ДАННЫХ -----------
DB_NAME = ""
DB_USER = ""
DB_PASSWORD = ""
DB_HOST = ""
DB_PORT = 5432

DB_URL = f"postgresql+pg8000://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()

log_filename = "camera1.log"
logging.basicConfig(
    filename=log_filename,
    filemode='a',
    format='%(asctime)s [%(levelname)s] %(message)s',
    level=logging.INFO,
    encoding='utf-8'
)

# Пример, чтобы убедиться
logging.info("🎬 Старт программы. Логгер инициализирован.")


class QueueAlert(Base):
    __tablename__ = "queue_alerts"
    id = Column(Integer, primary_key=True)
    camera = Column(String, default="Камера 1")
    sector = Column(String, default="A")
    zone = Column(String, nullable=False)
    timestamp = Column(TIMESTAMP, server_default=func.now())
    reason = Column(String, nullable=True)
    message = Column(String, nullable=False)

class QueueImage(Base):
    __tablename__ = "monitor_queueimage"
    id = Column(Integer, primary_key=True)
    camera = Column(String, default="Камера 1")
    sector = Column(String, default="A")
    zone = Column(String, nullable=False)
    image = Column(String, nullable=False)
    number_of_people = Column(Integer)
    timestamp = Column(TIMESTAMP, server_default=func.now())

Base.metadata.create_all(engine)

# ----------- ROI -----------
roi_data = np.load("queue_roi1full.npy", allow_pickle=True).item()
snake_roi = roi_data["snake"]
entry_roi = roi_data.get("entry", np.array([]))
exit_roi = roi_data.get("exit", np.array([]))

def is_inside_roi(point, roi):
    if roi.size == 0:
        return False
    return cv2.pointPolygonTest(roi, point, False) >= 0

# ----------- ФУНКЦИИ -----------
def save_alert(zone, reason=None):
    session = Session()
    alert = QueueAlert(zone=zone, reason = reason, message="Открыть новую стойку")
    session.add(alert)
    session.commit()
    session.close()
    logging.info(f"✅ Запись в БД: [{zone}] Открыть новую стойку | Причина: {reason}")

def save_screenshot(frame, zone, number_of_people):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{zone}_{timestamp}.jpg"  # только имя файла
    full_path = os.path.join(screenshot_dir, filename)  # полный путь для сохранения
    success = cv2.imwrite(full_path, frame)  # сохраняем по полному пути

    if success:
        session = Session()
        record = QueueImage(zone=zone, image=filename, number_of_people=number_of_people)  # сохраняем только имя файла в БД
        session.add(record)
        session.commit()
        session.close()
        logging.info(f"🖼 Скриншот сохранен: {full_path}, Людей: {number_of_people}")
    else:
        logging.info(f"❌ Не удалось сохранить скриншот: {full_path}")


# ----------- МОДЕЛЬ и ТРЕКЕР -----------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Используется устройство: {device}")
model = YOLO(model_path).to(device)
tracker = DeepSort(max_age=10)

# ----------- ОЧЕРЕДЬ КАДРОВ -----------
frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

# ----------- ЧТЕНИЕ RTSP через FFMPEG -----------
def read_rtsp():
    ffmpeg_cmd = [
        'ffmpeg',
        '-rtsp_transport', 'tcp',
        '-i', RTSP_URL,
        '-f', 'image2pipe',
        '-pix_fmt', 'bgr24',
        '-vcodec', 'rawvideo', '-'
    ]
    pipe = sp.Popen(ffmpeg_cmd, stdout=sp.PIPE, bufsize=10**8)

    while True:
        raw_image = pipe.stdout.read(FRAME_WIDTH * FRAME_HEIGHT * 3)
        if not raw_image:
            print("🔴 Поток остановился.")
            break
        try:
            frame = np.frombuffer(raw_image, dtype=np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3)).copy()
            if not frame_queue.full():
                frame_queue.put(frame)
        except Exception as e:
            print(f"⚠️ Ошибка при разборе кадра: {e}")
            continue

# ----------- ОБРАБОТКА КАДРОВ -----------
def get_open_counters_from_db():
    session = Session()
    try:
        # Получаем все названия стоек
        result = session.execute(text('SELECT stand_name FROM "Reception_desk_congestion"')).fetchall()
        open_counter_count = 0
        for row in result:
            stand_name = row[0]
            # Извлекаем номер из строки, например "Стойка регистрации №15" -> 15
            match = re.search(r'№(\d+)', stand_name)
            if match:
                number = int(match.group(1))
                if 15 <= number <= 22:
                    open_counter_count += 1
        return open_counter_count or 1  # По умолчанию 1, если ничего не нашли
    except Exception as e:
        logging.warning(f"Ошибка при запросе к стойкам: {e}")
        return 1
    finally:
        session.close()

def process_frames():
    people_tracker = {}
    passage_times = {}
    last_screenshot_time = time.time()
    alert_sent_snake = False
    frame_buffer = []
    snake_counts_buffer = []  # 🔁 для среднего количества людей в змейке
    minute_start_time = time.time()
    duration_log = []
    duration_update_interval = 10
    last_duration_update = time.time()
    zone_entry_counters = {
        "entry": {},
        "snake": {},
        "exit": {}
    }
    entries_5min = 0
    exits_5min = 0
    last_zone_summary_time = time.time()

    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3)

    while True:
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        current_time = time.time()
        results = model(frame)
        count_snake = 0
        detections = []
        people_in_queue = set()
        yolo_centers = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                if cls == 0 and conf > 0.4:
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, None))
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    yolo_centers.append((cx, cy))

        tracked_objects = tracker.update_tracks(detections, frame=frame)

        for track in tracked_objects:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = track.to_ltrb()
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            for zone_name, roi in [("entry", entry_roi), ("snake", snake_roi), ("exit", exit_roi)]:
                if is_inside_roi(center, roi):
                    if track.track_id not in zone_entry_counters[zone_name]:
                        zone_entry_counters[zone_name][track.track_id] = current_time
                        logging.info(f"🧾 Зашёл в зону '{zone_name}': ID {track.track_id}")
                        if zone_name == "snake":
                            count_snake += 1
                            people_in_queue.add(track.track_id)
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Для времени прохождения
            if is_inside_roi(center, entry_roi):
                if track.track_id not in passage_times:
                    passage_times[track.track_id] = {'entry': current_time, 'exit': None}
            elif is_inside_roi(center, exit_roi):
                if track.track_id in passage_times and passage_times[track.track_id]['exit'] is None:
                    passage_times[track.track_id]['exit'] = current_time
                    duration = current_time - passage_times[track.track_id]['entry']
                    duration_log.append((current_time, duration))
                    logging.warning(f"⏱ ID {track.track_id} прошёл очередь за {duration:.2f} секунд")


        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = pose_detector.process(rgb_frame)

        count_pose_people = 0
        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark
            xs = [l.x for l in landmarks]
            ys = [l.y for l in landmarks]
            h, w, _ = frame.shape
            cx = int(np.mean(xs) * w)
            cy = int(np.mean(ys) * h)
            pose_center = (cx, cy)
            duplicate = False
            for (x, y) in yolo_centers:
                if np.linalg.norm(np.array(pose_center) - np.array((x, y))) < 50:
                    duplicate = True
                    break
            if not duplicate and is_inside_roi(pose_center, snake_roi):
                count_snake += 1
                count_pose_people += 1
                cv2.circle(frame, pose_center, 10, (0, 0, 255), -1)

        if count_pose_people > 0:
            logging.info(f"🩻 MediaPipe добавил: {count_pose_people} человек(а)")

        for person_id in people_in_queue:
            if person_id not in people_tracker:
                people_tracker[person_id] = current_time

        stuck_people_snake = [p for p in people_tracker if (current_time - people_tracker[p]) > ALERT_TIME]
        if len(stuck_people_snake) >= 3 and not alert_sent_snake:
            save_alert("Snake Queue", reason="Много людей")
            alert_sent_snake = True

        people_tracker = {p: people_tracker[p] for p in people_in_queue}

        frame_buffer.append((frame.copy(), count_snake))
        snake_counts_buffer.append((current_time, count_snake))

        # Скриншот раз в минуту
        if current_time - minute_start_time >= 60:
            if frame_buffer:
                best_frame, max_count = max(frame_buffer, key=lambda x: x[1])
                save_screenshot(best_frame, "Snake Queue", max_count)
            frame_buffer.clear()
            minute_start_time = current_time

        # Очистка проходов
        passage_times = {
            tid: t for tid, t in passage_times.items()
            if t['exit'] is None or (current_time - t['exit']) < 600
        }

        # Обновление статистики каждые 10 сек
        # Обновление статистики каждые 5 минут
        if current_time - last_zone_summary_time >= 300:
            entries_5min = len(zone_entry_counters["entry"])
            exits_5min = len(zone_entry_counters["exit"])

            for zone_name, tracker_dict in zone_entry_counters.items():
                total_count = len(tracker_dict)
                logging.warning(f"📥 За 5 минут в зону '{zone_name}' зашли: {total_count} человек(а)")

            # Очистка счётчиков
            last_zone_summary_time = current_time

            # 🚨 Проверка на дисбаланс входов и выходов
            if exits_5min == 0 and entries_5min > 0:
                logging.warning(f"⚠️ Дисбаланс: есть входы, но нет выходов — генерируем алерт")
                save_alert("Snake Queue", reason="Люди входят и не выходят")
            elif exits_5min > 0:
                imbalance_ratio = (entries_5min - exits_5min) / exits_5min
                if imbalance_ratio > 0.2:
                    logging.warning(f"⚠️ Дисбаланс >20%: входов={entries_5min}, выходов={exits_5min}, разница={imbalance_ratio:.2%}")
                    save_alert("Snake Queue", reason="Людей больше зашло, чем вышло, более чем на 20%")
                else:
                    logging.info(f"✅ Баланс в норме: входов={entries_5min}, выходов={exits_5min}, разница={imbalance_ratio:.2%}")
            else:
                logging.info("✅ Нет входов и выходов — активность отсутствует")
            zone_entry_counters = {
                "entry": {},
                "snake": {},
                "exit": {}
            }
            # 📊 Среднее число людей в змейке за последние 5 минут
            open_counters = get_open_counters_from_db()
            max_allowed = open_counters * 5
            avg_snake_count = np.mean([count for _, count in snake_counts_buffer]) if snake_counts_buffer else 0
            snake_counts_buffer.clear()

            if avg_snake_count > max_allowed:
                logging.warning(f"⚠️ Перегрузка: {avg_snake_count:.1f} > {max_allowed}")
                save_alert("Snake Queue", reason="Перегруз людей, не хватает количества стоек")
            else:
                logging.info(f"✅ Змейка в норме: среднее количество {avg_snake_count:.1f} при допуске {max_allowed}")



        # Отрисовка
        cv2.polylines(frame, [snake_roi.astype(np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)
        if entry_roi.size:
            cv2.polylines(frame, [entry_roi.astype(np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)
        if exit_roi.size:
            cv2.polylines(frame, [exit_roi.astype(np.int32)], isClosed=True, color=(0, 165, 255), thickness=2)

        cv2.putText(frame, f'Snake Queue: {count_snake}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Queue Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

t1 = threading.Thread(target=read_rtsp, daemon=True)
t2 = threading.Thread(target=process_frames, daemon=True)

t1.start()
t2.start()

t1.join()
t2.join()
cv2.destroyAllWindows()