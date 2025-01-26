from django.contrib import admin
from django.urls import path,include
from django.conf.urls.static import static
from django.conf import settings


admin.site.site_header = "HOSPITAL MANAGEMENT"
admin.site.site_title = "POPSON ADMIN"

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('hospital.urls')),
    path('popauth/', include('popauth.urls')),
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)




  

