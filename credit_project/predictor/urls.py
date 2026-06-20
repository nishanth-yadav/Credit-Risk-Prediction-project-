from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing, name="landing"),
    path('home/', views.home, name="home"),
    path('register/', views.register, name="register"),
    path('login/', views.user_login, name="login"),
    path('logout/', views.user_logout, name="logout"),
    path('input/', views.input_page, name="input"),
    path('predict/', views.predict, name="predict"),
    path('explanation/', views.explanation, name="explanation"),
    path('about/', views.about_model, name="about"),
    path('train/', views.train_page, name="train_page"),
    path('start-training/', views.start_training, name="start_training"),
]
