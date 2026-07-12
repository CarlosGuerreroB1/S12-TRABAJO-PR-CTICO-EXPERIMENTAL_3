import cv2
import numpy as np
from django.conf import settings
import os

CV2_DATA = getattr(settings, 'CV2_DATA_PATH', cv2.data.haarcascades)


class ObjectDetector:

    CASCADES = {
        'face':       'haarcascade_frontalface_alt2.xml',
        'smile':      'haarcascade_smile.xml',
        'upper_body': 'haarcascade_upperbody.xml',
        'full_body':  'haarcascade_fullbody.xml',
        'eye':        'haarcascade_eye.xml',
    }

    def __init__(self, detection_type='face'):
        self.detection_type = detection_type
        self.cascades = {}
        for key, fname in self.CASCADES.items():
            path = os.path.join(CV2_DATA, fname)
            c = cv2.CascadeClassifier(path)
            if not c.empty():
                self.cascades[key] = c

    def detect(self, frame):
        method = getattr(self, f'_detect_{self.detection_type}', self._detect_face)
        frame, detections = method(frame)
        self._draw_overlay(frame, detections)
        return frame, detections

    def _detect_face(self, frame):
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        faces = self.cascades['face'].detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))
        detections = []
        for (x, y, w, h) in (faces if len(faces) else []):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 100), 2)
            self._label(frame, 'Rostro', x, y, (0, 255, 100))
            detections.append({'label': 'Rostro', 'confidence': 0.90,
                                'bbox': (int(x), int(y), int(w), int(h))})
        return frame, detections

    def _detect_mask(self, frame):
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        faces = self.cascades['face'].detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
        detections = []
        for (x, y, w, h) in (faces if len(faces) else []):
            face_gray = gray[y:y+h, x:x+w]
            smiles = self.cascades['smile'].detectMultiScale(
                face_gray, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25))
            if len(smiles) > 0:
                color = (0, 0, 255)
                label = 'Sin Mascarilla'
            else:
                color = (0, 255, 100)
                label = 'Con Mascarilla'
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            self._label(frame, label, x, y, color)
            detections.append({'label': label, 'confidence': 0.80,
                                'bbox': (int(x), int(y), int(w), int(h))})
        return frame, detections

    def _detect_helmet(self, frame):
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        bodies = self.cascades['upper_body'].detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(80, 80))
        detections = []
        for (x, y, w, h) in (bodies if len(bodies) else []):
            head_region = gray[y:y+h//2, x:x+w]
            faces = self.cascades['face'].detectMultiScale(
                head_region, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            if len(faces) == 0:
                color = (0, 200, 255)
                label = 'Casco Detectado'
            else:
                color = (0, 165, 255)
                label = 'Sin Casco'
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            self._label(frame, label, x, y, color)
            detections.append({'label': label, 'confidence': 0.75,
                                'bbox': (int(x), int(y), int(w), int(h))})
        return frame, detections

    def _detect_phone(self, frame):
        detections = []
        h_frame, w_frame = frame.shape[:2]
        area_frame = h_frame * w_frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Dos pasadas de Canny con distintos umbrales para capturar mas bordes
        edges1 = cv2.Canny(blurred, 20, 80)
        edges2 = cv2.Canny(blurred, 50, 150)
        edges = cv2.bitwise_or(edges1, edges2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Mas permisivo: desde 0.5% hasta 40% del frame
            if area < area_frame * 0.005 or area > area_frame * 0.40:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            if len(approx) < 4 or len(approx) > 8:
                continue
            x, y, w, h = cv2.boundingRect(approx)
            aspect = max(h, w) / (min(h, w) + 1e-5)
            # Acepta desde 1.3 hasta 4.0
            if aspect < 1.3 or aspect > 4.0:
                continue
            solidity = area / (w * h + 1e-5)
            if solidity < 0.45:
                continue
            score = solidity * min(aspect / 2.0, 1.0)
            candidates.append((score, x, y, w, h))

        candidates.sort(key=lambda c: c[0], reverse=True)
        kept = []
        for cand in candidates[:10]:
            score, x, y, w, h = cand
            overlap = False
            for _, kx, ky, kw, kh in kept:
                ix = max(x, kx); iy = max(y, ky)
                iw = min(x+w, kx+kw) - ix
                ih = min(y+h, ky+kh) - iy
                if iw > 0 and ih > 0:
                    inter = iw * ih
                    union = w*h + kw*kh - inter
                    if inter / union > 0.3:
                        overlap = True
                        break
            if not overlap:
                kept.append(cand)

        for score, x, y, w, h in kept[:3]:
            conf = min(0.55 + score * 0.3, 0.95)
            color = (255, 100, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            self._label(frame, f'Celular ({conf*100:.0f}%)', x, y, color)
            detections.append({'label': 'Celular', 'confidence': round(conf, 2),
                                'bbox': (int(x), int(y), int(w), int(h))})
        return frame, detections

    def _detect_body(self, frame):
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        bodies = self.cascades['full_body'].detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(60, 120))
        detections = []
        for (x, y, w, h) in (bodies if len(bodies) else []):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 165, 0), 2)
            self._label(frame, 'Persona', x, y, (255, 165, 0))
            detections.append({'label': 'Persona', 'confidence': 0.80,
                                'bbox': (int(x), int(y), int(w), int(h))})
        return frame, detections

    def _label(self, frame, text, x, y, color):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y0 = max(y - 5, th + 10)
        cv2.rectangle(frame, (x, y0 - th - 8), (x + tw + 8, y0 + 2), color, -1)
        cv2.putText(frame, text, (x + 4, y0 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    def _draw_overlay(self, frame, detections):
        labels = {
            'face': 'Deteccion de Rostros',
            'mask': 'Deteccion de Mascarillas',
            'helmet': 'Deteccion de Cascos',
            'phone': 'Deteccion de Celulares',
            'body': 'Deteccion de Personas',
        }
        title = labels.get(self.detection_type, 'Deteccion')
        count = len(detections)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (300, 65), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, title, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f'Detectados: {count}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 150), 2, cv2.LINE_AA)


class VideoStreamProcessor:
    def __init__(self, detection_type='face', camera_index=0):
        self.detection_type = detection_type
        self.camera_index = camera_index
        self.detector = ObjectDetector(detection_type)

    def generate_frames(self):
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            frame = self._error_frame()
            while True:
                _, jpeg = cv2.imencode('.jpg', frame)
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
            return
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame, _ = self.detector.detect(frame)
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
        finally:
            cap.release()

    @staticmethod
    def _error_frame():
        f = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(f, 'Camara no disponible', (80, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 80, 255), 2)
        return f
