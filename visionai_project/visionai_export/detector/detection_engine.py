"""
Motor de detección de objetos usando OpenCV.
Detecta: rostros, cuerpos, ojos (como proxies para mascarillas/cascos/celulares).
"""
import cv2
import numpy as np
from django.conf import settings
import os

CV2_DATA = getattr(settings, 'CV2_DATA_PATH', cv2.data.haarcascades)


class ObjectDetector:
    """Detector de objetos configurable basado en Haar Cascades de OpenCV."""

    DETECTORS = {
        'face': {
            'cascade': 'haarcascade_frontalface_alt2.xml',
            'label': 'Rostro',
            'color': (0, 255, 100),
            'scale': 1.1,
            'neighbors': 5,
            'min_size': (60, 60),
        },
        'body': {
            'cascade': 'haarcascade_fullbody.xml',
            'label': 'Persona',
            'color': (255, 165, 0),
            'scale': 1.05,
            'neighbors': 3,
            'min_size': (80, 160),
        },
        'upper_body': {
            'cascade': 'haarcascade_upperbody.xml',
            'label': 'Cuerpo Superior',
            'color': (0, 200, 255),
            'scale': 1.05,
            'neighbors': 3,
            'min_size': (80, 80),
        },
        'eye': {
            'cascade': 'haarcascade_eye.xml',
            'label': 'Ojo / Lentes',
            'color': (255, 50, 50),
            'scale': 1.1,
            'neighbors': 8,
            'min_size': (20, 20),
        },
        'smile': {
            'cascade': 'haarcascade_smile.xml',
            'label': 'Sin Mascarilla',
            'color': (0, 0, 255),
            'scale': 1.7,
            'neighbors': 20,
            'min_size': (25, 25),
        },
    }

    # Mapeo de tipos de detección de usuario → detectores activos
    TYPE_MAP = {
        'face':   ['face'],
        'mask':   ['face', 'smile'],   # detecta cara + sonrisa (sin mascarilla)
        'helmet': ['face', 'upper_body'],
        'phone':  ['face', 'eye'],
        'body':   ['body'],
    }

    def __init__(self, detection_type='face'):
        self.detection_type = detection_type
        self.cascades = {}
        self._load_cascades()

    def _load_cascades(self):
        active = self.TYPE_MAP.get(self.detection_type, ['face'])
        for key in active:
            cfg = self.DETECTORS[key]
            path = os.path.join(CV2_DATA, cfg['cascade'])
            cascade = cv2.CascadeClassifier(path)
            if cascade.empty():
                print(f"[WARN] No se pudo cargar: {path}")
            else:
                self.cascades[key] = (cascade, cfg)

    def detect(self, frame):
        """
        Procesa un frame y retorna:
          - frame anotado
          - lista de detecciones [{label, confidence, bbox}]
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        detections = []

        for key, (cascade, cfg) in self.cascades.items():
            objects = cascade.detectMultiScale(
                gray,
                scaleFactor=cfg['scale'],
                minNeighbors=cfg['neighbors'],
                minSize=cfg['min_size'],
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

            if len(objects) == 0:
                continue

            for (x, y, w, h) in objects:
                color = cfg['color']
                label = cfg['label']

                # Recuadro principal
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Fondo del texto
                text = label
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x, y - th - 10), (x + tw + 8, y), color, -1)

                # Texto
                cv2.putText(
                    frame, text,
                    (x + 4, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA
                )

                detections.append({
                    'label': label,
                    'confidence': 0.85,
                    'bbox': (int(x), int(y), int(w), int(h)),
                })

        # Overlay con info
        self._draw_overlay(frame, len(detections))
        return frame, detections

    def _draw_overlay(self, frame, count):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (280, 60), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        type_labels = {
            'face': 'Detección de Rostros',
            'mask': 'Detección de Mascarillas',
            'helmet': 'Detección de Cascos',
            'phone': 'Detección de Lentes/Ojos',
            'body': 'Detección de Personas',
        }
        title = type_labels.get(self.detection_type, 'Detección')
        cv2.putText(frame, title, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f'Detectados: {count}', (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 150), 2, cv2.LINE_AA)


class VideoStreamProcessor:
    """Generador de frames MJPEG para streaming HTTP."""

    def __init__(self, detection_type='face', camera_index=0):
        self.detection_type = detection_type
        self.camera_index = camera_index
        self.detector = ObjectDetector(detection_type)

    def generate_frames(self):
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            # Devuelve frame de error si no hay cámara
            error_frame = self._error_frame()
            while True:
                _, jpeg = cv2.imencode('.jpg', error_frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + jpeg.tobytes() + b'\r\n')
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame, _ = self.detector.detect(frame)
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + jpeg.tobytes() + b'\r\n')
        finally:
            cap.release()

    @staticmethod
    def _error_frame():
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, 'Camara no disponible', (80, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 80, 255), 2)
        cv2.putText(frame, 'Verifica permisos de camara', (60, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
        return frame
