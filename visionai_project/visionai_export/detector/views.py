import json
import base64
import cv2
import numpy as np
from datetime import timedelta
from django.shortcuts import render, get_object_or_404
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.http import require_GET
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone

from .models import DetectionSession, DetectionEvent
from .detection_engine import VideoStreamProcessor, ObjectDetector


def index(request):
    recent_sessions = DetectionSession.objects.order_by("-started_at")[:5]
    context = {
        "recent_sessions": recent_sessions,
        "total_sessions": DetectionSession.objects.count(),
        "total_detections": DetectionEvent.objects.count(),
        "active_sessions": DetectionSession.objects.filter(is_active=True).count(),
    }
    return render(request, "detector/index.html", context)


def live_view(request):
    return render(request, "detector/live.html")


def history(request):
    sessions = DetectionSession.objects.prefetch_related("events").order_by("-started_at")
    return render(request, "detector/history.html", {"sessions": sessions})


def session_detail(request, pk):
    session = get_object_or_404(DetectionSession, pk=pk)
    events = session.events.order_by("-detected_at")[:50]
    return render(request, "detector/session_detail.html", {"session": session, "events": events})


def video_feed(request):
    processor = VideoStreamProcessor(detection_type="phone")
    response = StreamingHttpResponse(processor.generate_frames(), content_type="multipart/x-mixed-replace; boundary=frame")
    response["Cache-Control"] = "no-cache"
    return response


@csrf_exempt
def start_session(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST requerido"}, status=405)
    DetectionSession.objects.filter(is_active=True).update(is_active=False, ended_at=timezone.now())
    session = DetectionSession.objects.create(detection_type="phone", is_active=True)
    return JsonResponse({"session_id": session.pk, "status": "started"})


@csrf_exempt
def stop_session(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST requerido"}, status=405)
    try:
        data = json.loads(request.body)
    except Exception:
        data = {}
    session_id = data.get("session_id")
    try:
        session = DetectionSession.objects.get(pk=session_id)
        session.is_active = False
        session.ended_at = timezone.now()
        session.save()
        return JsonResponse({"status": "stopped", "total": session.total_detections})
    except DetectionSession.DoesNotExist:
        return JsonResponse({"error": "No encontrada"}, status=404)


@csrf_exempt
def delete_session(request, pk):
    if request.method != "DELETE":
        return JsonResponse({"error": "DELETE requerido"}, status=405)
    try:
        session = DetectionSession.objects.get(pk=pk)
        session.delete()
        return JsonResponse({"status": "deleted"})
    except DetectionSession.DoesNotExist:
        return JsonResponse({"error": "No encontrada"}, status=404)


@csrf_exempt
def delete_all_sessions(request):
    if request.method != "DELETE":
        return JsonResponse({"error": "DELETE requerido"}, status=405)
    count, _ = DetectionSession.objects.all().delete()
    return JsonResponse({"status": "deleted", "count": count})


@csrf_exempt
def capture_snapshot(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST requerido"}, status=405)
    try:
        data = json.loads(request.body)
    except Exception:
        return JsonResponse({"error": "JSON invalido"}, status=400)
    image_data = data.get("image", "")
    session_id = data.get("session_id")
    if not image_data:
        return JsonResponse({"error": "Sin imagen"}, status=400)
    try:
        if "," in image_data:
            image_data = image_data.split(",")[1]
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)
    if frame is None:
        return JsonResponse({"error": "Frame invalido"}, status=400)
    detector = ObjectDetector("phone")
    annotated, detections = detector.detect(frame)
    if session_id and detections:
        try:
            session = DetectionSession.objects.get(pk=session_id, is_active=True)
            for det in detections:
                bbox = det.get("bbox", (0, 0, 0, 0))
                DetectionEvent.objects.create(
                    session=session, object_label=det["label"], confidence=det["confidence"],
                    bbox_x=bbox[0], bbox_y=bbox[1], bbox_w=bbox[2], bbox_h=bbox[3],
                )
            session.total_detections += len(detections)
            session.save(update_fields=["total_detections"])
        except DetectionSession.DoesNotExist:
            pass
    _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    result_b64 = base64.b64encode(jpeg.tobytes()).decode("utf-8")
    return JsonResponse({
        "annotated_image": f"data:image/jpeg;base64,{result_b64}",
        "detections": [{"label": d["label"], "confidence": d["confidence"], "bbox": list(d["bbox"])} for d in detections],
        "count": len(detections),
    })


@require_GET
def stats_api(request):
    last_7 = []
    today = timezone.now().date()
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        count = DetectionEvent.objects.filter(detected_at__date=day).count()
        last_7.append({"date": str(day), "count": count})
    return JsonResponse({
        "last_7_days": last_7,
        "total_sessions": DetectionSession.objects.count(),
        "total_detections": DetectionEvent.objects.count(),
    })
