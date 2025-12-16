import cv2
import numpy as np

def select_polygon_roi(frame, zone_name):
    points = []

    def draw_roi(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            if len(points) > 1:
                cv2.line(frame, points[-2], points[-1], (0, 255, 0), 2)
            cv2.imshow(f"Выделите зону {zone_name}", frame)
        elif event == cv2.EVENT_RBUTTONDOWN:
            print(f"✅ Зона '{zone_name}' выбрана.")
            cv2.destroyWindow(f"Выделите зону {zone_name}")

    clone = frame.copy()
    cv2.imshow(f"Выделите зону {zone_name}", clone)
    cv2.setMouseCallback(f"Выделите зону {zone_name}", draw_roi)
    cv2.waitKey(0)
    return np.array(points)

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Ошибка загрузки видео!")
        return

    print("🔺 Выберите зону 'snake' (ЛКМ — добавить точку, ПКМ — завершить)")
    snake_roi = select_polygon_roi(frame.copy(), "snake")

    print("🔶 Выберите зону 'entry' (ЛКМ — добавить точку, ПКМ — завершить)")
    entry_roi = select_polygon_roi(frame.copy(), "entry")

    print("🔷 Выберите зону 'exit' (ЛКМ — добавить точку, ПКМ — завершить)")
    exit_roi = select_polygon_roi(frame.copy(), "exit")

    cap.release()

    roi_dict = {
        "snake": snake_roi,
        "entry": entry_roi,
        "exit": exit_roi
    }
    np.save("queue_roi1full.npy", roi_dict)
    print("✅ Все зоны сохранены в 'queue_roi2full.npy'.")

if __name__ == "__main__":
    main("C:/Project/sdf.mp4")  # Укажи путь к нужному видео
