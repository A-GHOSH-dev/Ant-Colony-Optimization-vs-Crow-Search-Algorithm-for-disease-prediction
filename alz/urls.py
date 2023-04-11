from django.urls import path
from . import views
from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('acocsa', views.acocsa, name="acocsa"),
    path('heart', views.heart, name="heart"),
    path('breast', views.breast, name="breast"),
    path('', views.home, name="home"),
   
]

if settings.DEBUG:
        urlpatterns += static(settings.MEDIA_URL,
                              document_root=settings.MEDIA_ROOT)

