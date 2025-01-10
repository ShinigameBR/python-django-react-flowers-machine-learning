from django.apps import AppConfig
from django.conf import settings
import pandas as pd
from joblib import load
import os

class PredictionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'prediction'
    MLMODEL_FOLDER = os.path.join(settings.BASE_DIR, 'prediction/mlmodel/')
    MLMODEL_FILE = os.path.join(MLMODEL_FOLDER, "IRISRandomForestClassifier.joblib")
    mlmodel = load(MLMODEL_FILE)