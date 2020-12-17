from django.urls import path
from . import views

urlpatterns=[

    path('',views.blank,name='blank'),
    path('blank',views.blank,name='blank'),
    path('login',views.login,name='login'),
    path('main',views.main,name='main'),
    path('loadCSV',views.loadCSV,name='loadCSV'),
    path('detailsForm',views.detailsForm,name='detailsForm'),
    path('predict',views.predict,name='predict'),
    path('graph',views.graph,name='graph')
    # path('add',views.add,name='add')
]