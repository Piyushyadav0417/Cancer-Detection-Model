from django.urls import path
from .views import *

urlpatterns = [
    path('', home, name='home_page'),
    # path('', predict_view, name='predict_page'),
    path('model_1/', model_1, name='model1_page'),
    path('model_2/', model_2, name='model2_page'),
    # path('input/', input, name='input_page'),
    path('result/', result, name='result_page'),
    path('predict/', predict_view, name='predict_view'),
    path('clear-and-home/', clear_and_redirect_home, name='clear_and_home'),
]