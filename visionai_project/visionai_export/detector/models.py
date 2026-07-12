from django.db import models


class DetectionSession(models.Model):
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    detection_type = models.CharField(
        max_length=50,
        choices=[('phone', 'Celulares')],
        default='phone'
    )
    total_detections = models.PositiveIntegerField(default=0)
    is_active = models.BooleanField(default=True)

    class Meta:
        verbose_name = 'Sesión de Detección'
        verbose_name_plural = 'Sesiones de Detección'
        ordering = ['-started_at']

    def __str__(self):
        return f"Sesión #{self.pk} ({self.started_at:%d/%m/%Y %H:%M})"


class DetectionEvent(models.Model):
    session = models.ForeignKey(
        DetectionSession, on_delete=models.CASCADE, related_name='events'
    )
    detected_at = models.DateTimeField(auto_now_add=True)
    object_label = models.CharField(max_length=100)
    confidence = models.FloatField(default=0.0)
    bbox_x = models.IntegerField(default=0)
    bbox_y = models.IntegerField(default=0)
    bbox_w = models.IntegerField(default=0)
    bbox_h = models.IntegerField(default=0)
    snapshot = models.ImageField(upload_to='captures/', null=True, blank=True)

    class Meta:
        verbose_name = 'Evento de Detección'
        verbose_name_plural = 'Eventos de Detección'
        ordering = ['-detected_at']

    def __str__(self):
        return f"{self.object_label} @ {self.detected_at:%H:%M:%S}"


class DetectionStatistic(models.Model):
    date = models.DateField(auto_now_add=True, unique=True)
    total_sessions = models.PositiveIntegerField(default=0)
    total_detections = models.PositiveIntegerField(default=0)
    most_detected = models.CharField(max_length=100, blank=True)

    class Meta:
        verbose_name = 'Estadística'
        verbose_name_plural = 'Estadísticas'
        ordering = ['-date']

    def __str__(self):
        return f"Stats {self.date}"
