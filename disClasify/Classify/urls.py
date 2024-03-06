from django.urls import path
from . import views

urlpatterns = [
    path('', views.save_and_predict, name='upload_image'),
]
