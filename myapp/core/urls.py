from django.urls import path
from . import views
urlpatterns = [
    path("",views.index,name='index'),
    path("api/chat/",views.chatbot_response,name='chatbot_response'),
]
