from django.urls import path
from .views import (
    get_camera_1_data,
    get_camera_2_data,
    get_camera_3_data,
    get_camera_4_data,
    get_camera_5_data,
    get_alert_data1,
    get_alert_data2,
    get_alert_data3,
    get_reception_desk_status,
    resolve_alert,
    get_alerts2,
    home,
)

urlpatterns = [
    path('', home, name='home'),
    path('api/camera1/', get_camera_1_data, name='camera1_data'),
    path('api/camera2/', get_camera_2_data, name='camera2_data'),
    path('api/camera3/', get_camera_3_data, name='camera3_data'),
    path('api/camera4/', get_camera_4_data, name='camera4_data'),
    path('api/camera5/', get_camera_5_data, name='camera5_data'),
    path('api/alerts1/', get_alert_data1, name='alerts1_data'),
    path('api/alerts2/', get_alert_data2, name='alerts2_data'),
    path('api/alerts3/', get_alert_data3, name='alerts3_data'),
    path('api/reception_desks/', get_reception_desk_status, name='reception_desk_status'),
    path('api/resolve_alert/', resolve_alert, name='resolve_alert'),
    path('api/alert2/', get_alerts2, name='alerts2'),

]