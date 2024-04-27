from django.urls import path 
from . import views 
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [ 
    path('get-image', views.get_image, name='get-image'),
    path('about', views.about, name='about'), 
    path('', views.home, name='home'), 
]

if settings.DEBUG:
        urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)