from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView
from prediction.apps import PredictionConfig
import pandas as pd

# Create your views here.
# Class based view to predict based on IRIS model
@api_view()
class IRIS_Model_Predict(APIView):
    def post(self, request, format=None):
        features = self.extract_features(request.data)
        predicted_species = self.predict_species(features)
        response_dict = {"Predicted Iris Species": predicted_species}
        return Response(response_dict, status=200)

    def extract_features(self, data):
        """Extract features from the request data and convert them into a numpy array."""
        values = list(data.values())
        return pd.Series(values).to_numpy().reshape(1, -1)

    def predict_species(self, features):
        """Predict the Iris species based on the input features."""
        model = PredictionConfig.mlmodel
        y_pred = model.predict(features)
        target_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        predicted_species = pd.Series(y_pred).map(target_map).to_numpy()
        return predicted_species[0]