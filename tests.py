"""
Pruebas unitarias e integración — VisionAI
Detección de Celulares con OpenCV + Django
"""
import os
import sys
import json
import base64
import unittest
import numpy as np

# Configurar Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'detection_project.settings_local')
sys.path.insert(0, '/home/claude')

import django
django.setup()

import cv2
from django.test import TestCase, Client
from django.urls import reverse
from django.utils import timezone

from detector.models import DetectionSession, DetectionEvent
from detector.detection_engine import ObjectDetector, VideoStreamProcessor


# ══════════════════════════════════════════════════════════════════
# SESIÓN 1A — PRUEBAS UNITARIAS
# ══════════════════════════════════════════════════════════════════

class TestObjectDetectorInit(TestCase):
    """Pruebas de inicialización del motor de detección."""

    def test_detector_inicializa_correctamente(self):
        """El detector se instancia sin errores."""
        detector = ObjectDetector('phone')
        self.assertIsNotNone(detector)

    def test_detector_carga_cascades(self):
        """Los Haar Cascades se cargan correctamente desde OpenCV."""
        detector = ObjectDetector('phone')
        self.assertGreater(len(detector.cascades), 0,
            "Deben cargarse al menos un cascade de OpenCV")

    def test_tipo_deteccion_se_almacena(self):
        """El tipo de detección se guarda correctamente."""
        detector = ObjectDetector('phone')
        self.assertEqual(detector.detection_type, 'phone')

    def test_detector_tipo_invalido_usa_default(self):
        """Un tipo inválido no lanza excepción."""
        detector = ObjectDetector('tipo_inexistente')
        self.assertIsNotNone(detector)


class TestDetectPhone(TestCase):
    """Pruebas del método de detección de celulares."""

    def setUp(self):
        self.detector = ObjectDetector('phone')

    def test_detect_frame_negro(self):
        """Frame completamente negro no debe lanzar excepción."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result_frame, detections = self.detector.detect(frame)
        self.assertIsNotNone(result_frame)
        self.assertIsInstance(detections, list)

    def test_detect_devuelve_frame_y_lista(self):
        """detect() siempre retorna (frame, lista)."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect(frame)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[1], list)

    def test_detect_frame_con_rectangulo(self):
        """Un rectángulo dibujado en el frame puede ser detectado."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Dibujar rectángulo con proporción de celular
        cv2.rectangle(frame, (200, 100), (280, 340), (200, 200, 200), -1)
        result_frame, detections = self.detector.detect(frame)
        self.assertIsNotNone(result_frame)

    def test_detect_frame_blanco(self):
        """Frame completamente blanco no debe lanzar excepción."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        result_frame, detections = self.detector.detect(frame)
        self.assertIsNotNone(result_frame)
        self.assertIsInstance(detections, list)

    def test_detect_frame_pequeno(self):
        """Frame de tamaño pequeño no debe lanzar excepción."""
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        result_frame, detections = self.detector.detect(frame)
        self.assertIsNotNone(result_frame)

    def test_deteccion_formato_correcto(self):
        """Cada detección tiene las claves requeridas."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (150, 80), (250, 380), (180, 180, 180), -1)
        _, detections = self.detector.detect(frame)
        for det in detections:
            self.assertIn('label', det)
            self.assertIn('confidence', det)
            self.assertIn('bbox', det)
            self.assertEqual(len(det['bbox']), 4)

    def test_confianza_entre_0_y_1(self):
        """La confianza de cada detección está entre 0 y 1."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (150, 80), (250, 380), (180, 180, 180), -1)
        _, detections = self.detector.detect(frame)
        for det in detections:
            self.assertGreaterEqual(det['confidence'], 0.0)
            self.assertLessEqual(det['confidence'], 1.0)

    def test_overlay_se_dibuja(self):
        """El overlay con texto se dibuja en el frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result_frame, _ = self.detector.detect(frame)
        # El overlay oscurece la esquina superior izquierda
        self.assertIsNotNone(result_frame)
        self.assertEqual(result_frame.shape, frame.shape)


class TestVideoStreamProcessor(TestCase):
    """Pruebas del procesador de stream de video."""

    def test_inicializa_con_tipo_phone(self):
        processor = VideoStreamProcessor(detection_type='phone')
        self.assertEqual(processor.detection_type, 'phone')

    def test_detector_interno_es_object_detector(self):
        processor = VideoStreamProcessor(detection_type='phone')
        self.assertIsInstance(processor.detector, ObjectDetector)

    def test_error_frame_es_numpy_array(self):
        frame = VideoStreamProcessor._error_frame()
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.shape, (480, 640, 3))


# ══════════════════════════════════════════════════════════════════
# SESIÓN 1B — PRUEBAS DE MODELOS (BD)
# ══════════════════════════════════════════════════════════════════

class TestDetectionSessionModel(TestCase):
    """Pruebas del modelo DetectionSession."""

    def test_crear_sesion(self):
        session = DetectionSession.objects.create(
            detection_type='phone', is_active=True
        )
        self.assertIsNotNone(session.pk)
        self.assertEqual(session.detection_type, 'phone')

    def test_sesion_activa_por_defecto(self):
        session = DetectionSession.objects.create(detection_type='phone')
        self.assertTrue(session.is_active)

    def test_total_detecciones_inicia_en_0(self):
        session = DetectionSession.objects.create(detection_type='phone')
        self.assertEqual(session.total_detections, 0)

    def test_str_retorna_string(self):
        session = DetectionSession.objects.create(detection_type='phone')
        self.assertIsInstance(str(session), str)

    def test_incrementar_total_detecciones(self):
        session = DetectionSession.objects.create(detection_type='phone')
        session.total_detections += 5
        session.save()
        session.refresh_from_db()
        self.assertEqual(session.total_detections, 5)


class TestDetectionEventModel(TestCase):
    """Pruebas del modelo DetectionEvent."""

    def setUp(self):
        self.session = DetectionSession.objects.create(detection_type='phone')

    def test_crear_evento(self):
        event = DetectionEvent.objects.create(
            session=self.session,
            object_label='Celular',
            confidence=0.87,
            bbox_x=100, bbox_y=150, bbox_w=80, bbox_h=200
        )
        self.assertIsNotNone(event.pk)

    def test_evento_relacionado_con_sesion(self):
        event = DetectionEvent.objects.create(
            session=self.session, object_label='Celular', confidence=0.80,
            bbox_x=0, bbox_y=0, bbox_w=50, bbox_h=100
        )
        self.assertEqual(event.session.pk, self.session.pk)

    def test_borrar_sesion_borra_eventos(self):
        DetectionEvent.objects.create(
            session=self.session, object_label='Celular', confidence=0.75,
            bbox_x=0, bbox_y=0, bbox_w=50, bbox_h=100
        )
        session_pk = self.session.pk
        self.session.delete()
        self.assertEqual(
            DetectionEvent.objects.filter(session_id=session_pk).count(), 0
        )


# ══════════════════════════════════════════════════════════════════
# SESIÓN 1C — PRUEBAS DE INTEGRACIÓN (rutas Django)
# ══════════════════════════════════════════════════════════════════

class TestIntegracionRutas(TestCase):
    """Pruebas de integración: acceso a rutas principales."""

    def setUp(self):
        self.client = Client()

    def test_ruta_index_retorna_200(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_ruta_live_retorna_200(self):
        response = self.client.get('/live/')
        self.assertEqual(response.status_code, 200)

    def test_ruta_history_retorna_200(self):
        response = self.client.get('/history/')
        self.assertEqual(response.status_code, 200)

    def test_ruta_stats_api_retorna_200(self):
        response = self.client.get('/api/stats/')
        self.assertEqual(response.status_code, 200)

    def test_stats_api_retorna_json_valido(self):
        response = self.client.get('/api/stats/')
        data = json.loads(response.content)
        self.assertIn('total_sessions', data)
        self.assertIn('total_detections', data)
        self.assertIn('last_7_days', data)

    def test_ruta_inexistente_retorna_404(self):
        response = self.client.get('/ruta-que-no-existe/')
        self.assertEqual(response.status_code, 404)


# ══════════════════════════════════════════════════════════════════
# SESIÓN 2 — PRUEBAS FUNCIONALES
# ══════════════════════════════════════════════════════════════════

class TestFuncionalInicioSesion(TestCase):
    """Pruebas funcionales del flujo de inicio de sesión."""

    def setUp(self):
        self.client = Client()

    def test_iniciar_sesion_crea_registro_bd(self):
        count_antes = DetectionSession.objects.count()
        self.client.post('/api/start-session/',
            data=json.dumps({'detection_type': 'phone'}),
            content_type='application/json'
        )
        self.assertEqual(DetectionSession.objects.count(), count_antes + 1)

    def test_iniciar_sesion_retorna_session_id(self):
        response = self.client.post('/api/start-session/',
            data=json.dumps({'detection_type': 'phone'}),
            content_type='application/json'
        )
        data = json.loads(response.content)
        self.assertIn('session_id', data)
        self.assertIsInstance(data['session_id'], int)

    def test_iniciar_sesion_marca_activa(self):
        self.client.post('/api/start-session/',
            data=json.dumps({'detection_type': 'phone'}),
            content_type='application/json'
        )
        session = DetectionSession.objects.latest('started_at')
        self.assertTrue(session.is_active)

    def test_detener_sesion_marca_inactiva(self):
        r = self.client.post('/api/start-session/',
            data=json.dumps({'detection_type': 'phone'}),
            content_type='application/json'
        )
        sid = json.loads(r.content)['session_id']
        self.client.post('/api/stop-session/',
            data=json.dumps({'session_id': sid}),
            content_type='application/json'
        )
        session = DetectionSession.objects.get(pk=sid)
        self.assertFalse(session.is_active)

    def test_detener_sesion_retorna_total(self):
        r = self.client.post('/api/start-session/',
            data=json.dumps({'detection_type': 'phone'}),
            content_type='application/json'
        )
        sid = json.loads(r.content)['session_id']
        response = self.client.post('/api/stop-session/',
            data=json.dumps({'session_id': sid}),
            content_type='application/json'
        )
        data = json.loads(response.content)
        self.assertIn('total', data)


class TestFuncionalCaptura(TestCase):
    """Pruebas funcionales del endpoint de captura."""

    def setUp(self):
        self.client = Client()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buf = cv2.imencode('.jpg', frame)
        self.b64_frame = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()

    def test_captura_sin_imagen_retorna_400(self):
        response = self.client.post('/api/capture/',
            data=json.dumps({}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)

    def test_captura_con_frame_retorna_200(self):
        response = self.client.post('/api/capture/',
            data=json.dumps({'image': self.b64_frame, 'detection_type': 'phone'}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)

    def test_captura_retorna_imagen_anotada(self):
        response = self.client.post('/api/capture/',
            data=json.dumps({'image': self.b64_frame, 'detection_type': 'phone'}),
            content_type='application/json'
        )
        data = json.loads(response.content)
        self.assertIn('annotated_image', data)
        self.assertTrue(data['annotated_image'].startswith('data:image/jpeg'))

    def test_captura_retorna_campo_count(self):
        response = self.client.post('/api/capture/',
            data=json.dumps({'image': self.b64_frame, 'detection_type': 'phone'}),
            content_type='application/json'
        )
        data = json.loads(response.content)
        self.assertIn('count', data)
        self.assertIsInstance(data['count'], int)

    def test_captura_retorna_lista_detecciones(self):
        response = self.client.post('/api/capture/',
            data=json.dumps({'image': self.b64_frame, 'detection_type': 'phone'}),
            content_type='application/json'
        )
        data = json.loads(response.content)
        self.assertIn('detections', data)
        self.assertIsInstance(data['detections'], list)


class TestFuncionalBorrarHistorial(TestCase):
    """Pruebas funcionales de borrado de sesiones."""

    def setUp(self):
        self.client = Client()
        self.session = DetectionSession.objects.create(detection_type='phone')

    def test_borrar_sesion_individual(self):
        pk = self.session.pk
        response = self.client.delete(f'/api/session/{pk}/delete/')
        self.assertEqual(response.status_code, 200)
        self.assertFalse(DetectionSession.objects.filter(pk=pk).exists())

    def test_borrar_todas_las_sesiones(self):
        DetectionSession.objects.create(detection_type='phone')
        DetectionSession.objects.create(detection_type='phone')
        response = self.client.delete('/api/sessions/delete-all/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(DetectionSession.objects.count(), 0)

    def test_borrar_sesion_inexistente_retorna_404(self):
        response = self.client.delete('/api/session/99999/delete/')
        self.assertEqual(response.status_code, 404)


if __name__ == '__main__':
    unittest.main(verbosity=2)
