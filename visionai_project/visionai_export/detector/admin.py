from django.contrib import admin
from .models import DetectionSession, DetectionEvent, DetectionStatistic


@admin.register(DetectionSession)
class DetectionSessionAdmin(admin.ModelAdmin):
    list_display = ['pk', 'detection_type', 'started_at', 'total_detections', 'is_active']
    list_filter = ['detection_type', 'is_active']
    readonly_fields = ['started_at', 'ended_at', 'total_detections']


@admin.register(DetectionEvent)
class DetectionEventAdmin(admin.ModelAdmin):
    list_display = ['pk', 'session', 'object_label', 'confidence', 'detected_at']
    list_filter = ['object_label']
    readonly_fields = ['detected_at']


@admin.register(DetectionStatistic)
class DetectionStatisticAdmin(admin.ModelAdmin):
    list_display = ['date', 'total_sessions', 'total_detections', 'most_detected']
    readonly_fields = ['date']
