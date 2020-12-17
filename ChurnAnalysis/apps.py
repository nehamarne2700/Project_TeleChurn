from django.apps import AppConfig
from django.conf import settings
import pickle
import os

class ChurnanalysisConfig(AppConfig):
    name = 'ChurnAnalysis'
    path=os.path.join(settings.MODELS,'models.p')

    with open(path,'rb') as pickled:
        data=pickle.load(pickled)

    model=data['model']
