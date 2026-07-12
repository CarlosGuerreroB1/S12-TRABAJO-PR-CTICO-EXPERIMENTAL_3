# VisionAI — Detección de Objetos en Tiempo Real

Aplicativo web desarrollado con **Django + OpenCV** para detección de objetos en tiempo real mediante la cámara web.

## Características

- 🎥 **Stream MJPEG** en tiempo real desde el servidor
- 📸 **Captura desde navegador** con envío al servidor para análisis
- 🧠 **Detección con Haar Cascades** de OpenCV:
  - Rostros (`haarcascade_frontalface_alt2`)
  - Mascarillas (cara + boca sin cubrir via `haarcascade_smile`)
  - Cascos (cara + cuerpo superior)
  - Celulares/Ojos (`haarcascade_eye`)
  - Personas completas (`haarcascade_fullbody`)
- 📊 **Dashboard** con estadísticas y gráficas de los últimos 7 días
- 🗃️ **Historial** de sesiones y eventos almacenados en PostgreSQL
- 🎨 Interfaz oscura moderna y responsiva

## Requisitos

- Python 3.10+
- PostgreSQL 13+
- Cámara web (opcional para modo stream del servidor)

## Instalación

```bash
# 1. Clonar repositorio
git clone <tu-repositorio>
cd detection_project

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar base de datos PostgreSQL
# Crear la base de datos:
psql -U postgres -c "CREATE DATABASE detection_db;"

# 4. Variables de entorno
cp .env.example .env
# Editar .env con tus credenciales

# 5. Migraciones
python manage.py migrate

# 6. Crear superusuario (opcional)
python manage.py createsuperuser

# 7. Ejecutar servidor
python manage.py runserver
```

Abrir: http://127.0.0.1:8000

## Configuración de PostgreSQL

Editar `detection_project/settings.py` o usar variables de entorno:

```
DB_NAME=detection_db
DB_USER=postgres
DB_PASSWORD=tu_password
DB_HOST=localhost
DB_PORT=5432
```

## Estructura del proyecto

```
detection_project/          ← Configuración Django
detector/
  ├── detection_engine.py   ← Motor OpenCV (ObjectDetector, VideoStreamProcessor)
  ├── models.py             ← DetectionSession, DetectionEvent, DetectionStatistic
  ├── views.py              ← Vistas + API REST
  ├── urls.py               ← Rutas
  ├── admin.py              ← Panel de administración
  └── templates/detector/   ← HTML (base, index, live, history, session_detail)
media/captures/             ← Imágenes capturadas
requirements.txt
```

## Endpoints principales

| Ruta | Descripción |
|------|-------------|
| `/` | Dashboard principal |
| `/live/?type=face` | Vista en vivo (face/mask/helmet/phone/body) |
| `/history/` | Historial de sesiones |
| `/video-feed/` | Stream MJPEG |
| `/api/start-session/` | POST — Iniciar sesión |
| `/api/stop-session/` | POST — Detener sesión |
| `/api/capture/` | POST — Procesar frame base64 |
| `/api/stats/` | GET — Estadísticas JSON |
| `/admin/` | Panel de administración Django |

## Modos de detección

| Tipo | Cascades activos | Uso |
|------|-----------------|-----|
| `face` | frontalface_alt2 | Detección de rostros |
| `mask` | frontalface + smile | Detecta cara y si la boca está visible (sin mascarilla) |
| `helmet` | frontalface + upperbody | Detecta persona con/sin casco |
| `phone` | frontalface + eye | Detecta ojos/lentes |
| `body` | fullbody | Detección de personas completas |

## Sesiones 1, 2 y 3 cubiertas

- **Sesión 1**: Proyecto Django configurado, streaming de video en tiempo real implementado.
- **Sesión 2**: OpenCV integrado con múltiples detectores Haar Cascade. Resultados anotados sobre el video.
- **Sesión 3**: Interfaz web completa con dashboard, historial, estadísticas. Dos modos de captura (stream + navegador).
