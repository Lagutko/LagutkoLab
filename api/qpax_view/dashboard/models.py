from django.db import models

class QueueImage(models.Model):
    camera = models.CharField(max_length=255, default="Camera2")
    sector = models.CharField(max_length=255, default="A")
    zone = models.CharField(max_length=255)
    image = models.ImageField(upload_to='', blank=True, null=False)
    number_of_people = models.IntegerField()
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'monitor_queueimage'  # явно указываем имя таблицы

class QueueAlert(models.Model):
    camera = models.CharField(max_length=255, default="Camera1")
    sector = models.CharField(max_length=255, default="A") 
    zone = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True)
    reason = models.CharField(max_length=255, null=True, blank=True)
    message = models.TextField()
    is_resolved = models.BooleanField(default=False)  # <-- новый флаг

    class Meta:
        db_table = 'queue_alerts'

class ReceptionDeskCongestion(models.Model):
    id=models.AutoField(primary_key=True)
    check_in_start = models.DateTimeField()
    check_in_end = models.DateTimeField()
    stand_name = models.CharField(max_length=255)

    class Meta:
        managed = False  # Django не будет создавать или менять эту таблицу
        db_table = 'Reception_desk_congestion'  # имя существующей таблицы
