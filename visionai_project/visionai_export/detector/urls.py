from django.urls import path
from . import views

app_name = 'detector'

urlpatterns = [
    path('', views.index, name='index'),
    path('live/', views.live_view, name='live'),
    path('history/', views.history, name='history'),
    path('history/<int:pk>/', views.session_detail, name='session_detail'),
    path('video-feed/', views.video_feed, name='video_feed'),
    path('api/start-session/', views.start_session, name='start_session'),
    path('api/stop-session/', views.stop_session, name='stop_session'),
    path('api/capture/', views.capture_snapshot, name='capture_snapshot'),
    path('api/stats/', views.stats_api, name='stats_api'),
    path('api/session/<int:pk>/delete/', views.delete_session, name='delete_session'),
    path('api/sessions/delete-all/', views.delete_all_sessions, name='delete_all_sessions'),
]
