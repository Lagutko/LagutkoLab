from django.shortcuts import render
from django.http import JsonResponse
from .models import QueueImage, QueueAlert
import pytz
from django.utils import timezone
from .models import ReceptionDeskCongestion
from django.views.decorators.http import require_POST

def home(request):
    data = {
        'cameras': {'count': 28, 'total': 30, 'icon': '📷'},
        'registration': {'count': 20, 'total': 25, 'icon': '🛃'},
        'passport': {'count': 4, 'total': 5, 'icon': '🛂'}
    }
    return render(request, 'dashboard/home.html', {'data': data})

def get_camera_1_data(request):
    moskow_tz = pytz.timezone("Europe/Moscow")
    data_qs = QueueImage.objects.filter(camera='Camera1').order_by('-timestamp')  # без сортировки по времени
    if data_qs.exists():
        data_list = []
        for item in data_qs:
            local_time = item.timestamp.astimezone(moskow_tz)
            data_list.append({
                'sector': item.sector,
                'zone': item.zone,
                'image': item.image.url if item.image else None,
                'number_of_people': item.number_of_people,
                'timestamp': local_time.strftime('%Y-%m-%d %H:%M:%S'),
            })
        return JsonResponse(data_list, safe=False)  # важно: safe=False позволяет вернуть список
    return JsonResponse({'error': 'Нет данных по Камере 1'}, status=404)


def get_camera_2_data(request):
    moskow_tz = pytz.timezone("Europe/Moscow")
    data_qs = QueueImage.objects.filter(camera='Camera2').order_by('-timestamp')  # без сортировки по времени
    if data_qs.exists():
        data_list = []
        for item in data_qs:
            local_time = item.timestamp.astimezone(moskow_tz)
            data_list.append({
                'sector': item.sector,
                'zone': item.zone,
                'image': item.image.url if item.image else None,
                'number_of_people': item.number_of_people,
                'timestamp': local_time.strftime('%Y-%m-%d %H:%M:%S'),
            })
        return JsonResponse(data_list, safe=False)  # важно: safe=False позволяет вернуть список
    return JsonResponse({'error': 'Нет данных по Камере 2'}, status=404)

def get_camera_3_data(request):
    moskow_tz = pytz.timezone("Europe/Moscow")
    data_qs = QueueImage.objects.filter(camera='Camera3').order_by('-timestamp')  # без сортировки по времени
    if data_qs.exists():
        data_list = []
        for item in data_qs:
            local_time = item.timestamp.astimezone(moskow_tz)
            data_list.append({
                'sector': item.sector,
                'zone': item.zone,
                'image': item.image.url if item.image else None,
                'number_of_people': item.number_of_people,
                'timestamp': local_time.strftime('%Y-%m-%d %H:%M:%S'),
            })
        return JsonResponse(data_list, safe=False)  # важно: safe=False позволяет вернуть список
    return JsonResponse({'error': 'Нет данных по Камере 3'}, status=404)


def get_camera_4_data(request):
    moskow_tz = pytz.timezone("Europe/Moscow")
    data_qs = QueueImage.objects.filter(camera='Camera4').order_by('-timestamp')
    if data_qs.exists():
        data_list = []
        for item in data_qs:
            local_time = item.timestamp.astimezone(moskow_tz)
            data_list.append({
                'sector': item.sector,
                'zone': item.zone,
                'image': item.image.url if item.image else None,
                'number_of_people': item.number_of_people,
                'timestamp': local_time.strftime('%Y-%m-%d %H:%M:%S'),
            })
        return JsonResponse(data_list, safe=False)
    return JsonResponse({'error': 'Нет данных по Камере 4'}, status=404)


def get_camera_5_data(request):
    moskow_tz = pytz.timezone("Europe/Moscow")
    data_qs = QueueImage.objects.filter(camera='Camera5').order_by('-timestamp')
    if data_qs.exists():
        data_list = []
        for item in data_qs:
            local_time = item.timestamp.astimezone(moskow_tz)
            data_list.append({
                'sector': item.sector,
                'zone': item.zone,
                'image': item.image.url if item.image else None,
                'number_of_people': item.number_of_people,
                'timestamp': local_time.strftime('%Y-%m-%d %H:%M:%S'),
            })
        return JsonResponse(data_list, safe=False)
    return JsonResponse({'error': 'Нет данных по Камере 5'}, status=404)

def get_alert_data1(request):
    moskow_tz = pytz.timezone("Europe/Moscow")
    alerts = QueueAlert.objects.filter(camera='Camera1')  # без сортировки по времени
    if alerts.exists():
        data_list = []
        for alert in alerts:
            local_time = alert.timestamp.astimezone(moskow_tz)
            data_list.append({
                'sector': alert.sector,
                'zone': alert.zone,
                'reason': alert.reason,
                'message': alert.message,
                'timestamp': local_time.strftime('%Y-%m-%d %H:%M:%S'),
            })
        return JsonResponse(data_list, safe=False)  # важно: safe=False позволяет вернуть список
    return JsonResponse({'error': 'Нет данных по Камере 1'}, status=404)

def get_alert_data2(request):
    moskow_tz = pytz.timezone("Europe/Moscow")
    alerts = QueueAlert.objects.filter(camera='Camera2')  # без сортировки по времени
    if alerts.exists():
        data_list = []
        for alert in alerts:
            local_time = alert.timestamp.astimezone(moskow_tz)
            data_list.append({
                'sector': alert.sector,
                'zone': alert.zone,
                'reason': alert.reason,
                'message': alert.message,
                'timestamp': local_time.strftime('%Y-%m-%d %H:%M:%S'),
            })
        return JsonResponse(data_list, safe=False)  # важно: safe=False позволяет вернуть список
    return JsonResponse({'error': 'Нет данных по Камере 2'}, status=404)

def get_alert_data3(request):
    moskow_tz = pytz.timezone("Europe/Moscow")
    alerts = QueueAlert.objects.filter(camera='Camera3')  # без сортировки по времени
    if alerts.exists():
        data_list = []
        for alert in alerts:
            local_time = alert.timestamp.astimezone(moskow_tz)
            data_list.append({
                'sector': alert.sector,
                'zone': alert.zone,
                'reason': alert.reason,
                'message': alert.message,
                'timestamp': local_time.strftime('%Y-%m-%d %H:%M:%S'),
            })
        return JsonResponse(data_list, safe=False)  # важно: safe=False позволяет вернуть список
    return JsonResponse({'error': 'Нет данных по Камере 1'}, status=404)

def get_reception_desk_status(request):
    moskow_tz = pytz.timezone("Europe/Moscow")

    # Получаем все уникальные стойки
    stand_names = ReceptionDeskCongestion.objects.values_list('stand_name', flat=True).distinct()

    data = []

    for stand in stand_names:
        # Берем самую последнюю запись по TIMESTAMP
        d = ReceptionDeskCongestion.objects.filter(stand_name=stand).order_by('-check_in_start').first()
        if not d:
            continue

        start_time = d.check_in_start
        end_time = d.check_in_end

        # Если datetime "naive", делаем aware для текущей зоны
        if start_time and timezone.is_naive(start_time):
            start_time = timezone.make_aware(start_time, timezone=moskow_tz)
        if end_time and timezone.is_naive(end_time):
            end_time = timezone.make_aware(end_time, timezone=moskow_tz)

        # Стойка открыта если start есть и end нет
        is_open = bool(start_time and not end_time)

        data.append({
            'stand_name': d.stand_name,
            'check_in_start': start_time.strftime('%Y-%m-%d %H:%M') if start_time else '',
            'check_in_end': end_time.strftime('%Y-%m-%d %H:%M') if end_time else '',
            'is_open': is_open,
        })

    return JsonResponse(data, safe=False)

@require_POST
def resolve_alert(request):
    alert_id = request.POST.get('id')
    if not alert_id:
        return JsonResponse({'error': 'Нет ID алерта'}, status=400)
    
    try:
        alert = QueueAlert.objects.get(id=alert_id)
        alert.is_resolved = True
        alert.save()
        return JsonResponse({'status': 'ok'})
    except QueueAlert.DoesNotExist:
        return JsonResponse({'error': 'Алерт не найден'}, status=404)
    
def get_alerts2(request):
    moskow_tz = pytz.timezone("Europe/Moscow")
    alerts = QueueAlert.objects.filter(is_resolved=False)  # только не исправленные

    data_list = []
    for alert in alerts:
        local_time = alert.timestamp.astimezone(moskow_tz)
        data_list.append({
            'id': alert.id,
            'camera': alert.camera,
            'sector': alert.sector,
            'zone': alert.zone,
            'reason': alert.reason,
            'message': alert.message,
            'timestamp': local_time.strftime('%Y-%m-%d %H:%M:%S'),
        })

    return JsonResponse(data_list, safe=False)
