import json
import base64
import cv2
import numpy as np
from datetime import datetime, timedelta
from django.shortcuts import render, redirect, get_object_or_404
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.http import require_POST, require_GET
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone

from .models import DetectionSession, DetectionEvent, DetectionStatistic
from .detection_engine import VideoStreamProcessor, ObjectDetector


# ─── PÁGINAS ────────────────────────────────────────────────────────────────

def index(request):
    recent_sessions = DetectionSession.objects.order_by('-started_at')[:5]
    total_sessions = DetectionSession.objects.count()
    total_detections = DetectionEvent.objects.count()
    active_sessions = DetectionSession.objects.filter(is_active=True).count()

    context = {
        'recent_sessions': recent_sessions,
        'total_sessions': total_sessions,
        'total_detections': total_detections,
        'active_sessions': active_sessions,
        'detection_types': DetectionSession._meta.get_field('detection_type').choices,
    }
    return render(request, 'detector/index.html', context)


def live_view(request):
    detection_type = request.GET.get('type', 'face')
    session_id = request.GET.get('session_id')

    session = None
    if session_id:
        session = get_object_or_404(DetectionSession, pk=session_id)

    context = {
        'detection_type': detection_type,
        'session': session,
        'detection_types': DetectionSession._meta.get_field('detection_type').choices,
        'type_labels': {
            'face': 'Detección de Rostros',
            'mask': 'Detección de Mascarillas',
            'helmet': 'Detección de Cascos',
            'phone': 'Detección de Celulares / Lentes',
            'body': 'Detección de Personas',
        },
    }
    return render(request, 'detector/live.html', context)


def history(request):
    sessions = DetectionSession.objects.prefetch_related('events').order_by('-started_at')
    context = {'sessions': sessions}
    return render(request, 'detector/history.html', context)


def session_detail(request, pk):
    session = get_object_or_404(DetectionSession, pk=pk)
    events = session.events.order_by('-detected_at')[:50]
    context = {'session': session, 'events': events}
    return render(request, 'detector/session_detail.html', context)


# ─── API / STREAMING ────────────────────────────────────────────────────────

def video_feed(request):
    detection_type = request.GET.get('type', 'face')
    processor = VideoStreamProcessor(detection_type=detection_type)
    response = StreamingHttpResponse(
        processor.generate_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )
    response['Cache-Control'] = 'no-cache'
    return response


@require_POST
def start_session(request):
    data = json.loads(request.body)
    detection_type = data.get('detection_type', 'face')

    # Cerrar sesiones activas anteriores
    DetectionSession.objects.filter(is_active=True).update(
        is_active=False, ended_at=timezone.now()
    )

    session = DetectionSession.objects.create(
        detection_type=detection_type,
        is_active=True,
    )
    return JsonResponse({'session_id': session.pk, 'status': 'started'})


@require_POST
def stop_session(request):
    data = json.loads(request.body)
    session_id = data.get('session_id')
    try:
        session = DetectionSession.objects.get(pk=session_id)
        session.is_active = False
        session.ended_at = timezone.now()
        session.save()
        return JsonResponse({'status': 'stopped', 'total': session.total_detections})
    except DetectionSession.DoesNotExist:
        return JsonResponse({'error': 'Sesión no encontrada'}, status=404)


@csrf_exempt
@require_POST
def log_detection(request):
    """Registra una detección desde el cliente."""
    data = json.loads(request.body)
    session_id = data.get('session_id')
    detections = data.get('detections', [])

    try:
        session = DetectionSession.objects.get(pk=session_id, is_active=True)
    except DetectionSession.DoesNotExist:
        return JsonResponse({'error': 'Sesión no activa'}, status=400)

    for det in detections:
        bbox = det.get('bbox', [0, 0, 0, 0])
        DetectionEvent.objects.create(
            session=session,
            object_label=det.get('label', 'Objeto'),
            confidence=det.get('confidence', 0.0),
            bbox_x=bbox[0] if len(bbox) > 0 else 0,
            bbox_y=bbox[1] if len(bbox) > 1 else 0,
            bbox_w=bbox[2] if len(bbox) > 2 else 0,
            bbox_h=bbox[3] if len(bbox) > 3 else 0,
        )

    session.total_detections += len(detections)
    session.save(update_fields=['total_detections'])

    return JsonResponse({'logged': len(detections)})


@require_GET
def stats_api(request):
    """Estadísticas en tiempo real para el dashboard."""
    last_7 = []
    today = timezone.now().date()
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        count = DetectionEvent.objects.filter(detected_at__date=day).count()
        last_7.append({'date': str(day), 'count': count})

    by_type = {}
    for session in DetectionSession.objects.all():
        t = session.get_detection_type_display()
        by_type[t] = by_type.get(t, 0) + session.total_detections

    return JsonResponse({
        'last_7_days': last_7,
        'by_type': by_type,
        'total_sessions': DetectionSession.objects.count(),
        'total_detections': DetectionEvent.objects.count(),
    })


@csrf_exempt
@require_POST
def capture_snapshot(request):
    """Procesa un frame base64 del cliente y devuelve detecciones."""
    data = json.loads(request.body)
    image_data = data.get('image', '')
    detection_type = data.get('detection_type', 'face')
    session_id = data.get('session_id')

    # Decodificar imagen base64
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    img_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return JsonResponse({'error': 'Frame inválido'}, status=400)

    detector = ObjectDetector(detection_type)
    annotated, detections = detector.detect(frame)

    # Guardar en BD si hay sesión activa
    if session_id and detections:
        try:
            session = DetectionSession.objects.get(pk=session_id, is_active=True)
            for det in detections:
                bbox = det.get('bbox', (0, 0, 0, 0))
                DetectionEvent.objects.create(
                    session=session,
                    object_label=det['label'],
                    confidence=det['confidence'],
                    bbox_x=bbox[0], bbox_y=bbox[1],
                    bbox_w=bbox[2], bbox_h=bbox[3],
                )
            session.total_detections += len(detections)
            session.save(update_fields=['total_detections'])
        except DetectionSession.DoesNotExist:
            pass

    # Convertir frame anotado a base64
    _, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    result_b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')

    return JsonResponse({
        'annotated_image': f'data:image/jpeg;base64,{result_b64}',
        'detections': detections,
        'count': len(detections),
    })
