
import argparse
import json
import pandas as pd
from model import Model
from preprocessor import Preprocessor

class Pipeline:
    def __init__(self):
        self.model = Model()
        self.preprocessor = Preprocessor()

    def run(self, data_path, test=False):
        if test:
            # Load preprocessor and model for testing
            with open('preprocessor.json', 'r') as f:
                preprocessor_params = json.load(f)
            self.preprocessor = Preprocessor(**preprocessor_params)
            self.preprocessor.scaler.with_mean = False  # Fix for a bug in sklearn

            with open('model.json', 'r') as f:
                model_params = json.load(f)
            self.model = Model(**model_params)

            # Load test data and preprocess
            X_test = pd.read_csv(data_path)
            X_test = self.preprocessor.transform(X_test)

            # Make predictions and save to file
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            threshold = 0.5  # Change this threshold if desired
            y_pred = (y_pred_proba > threshold).astype(int)
            results = {'predict_probas': y_pred_proba.tolist(), 'threshold': threshold}
            with open('predictions.json', 'w') as f:
                json.dump(results, f)
        else:
            # Load train data and preprocess
            data = pd.read_csv(data_path)
            X = data.drop('target', axis=1)
            y = data['target']
            self.preprocessor.fit(X)
            X = self.preprocessor.transform(X)

            # Fit model and save to file
            self.model.fit(X, y)
            with open('preprocessor.json', 'w') as f:
                json.dump(self.preprocessor.__dict__, f)
            with open('model.json', 'w') as f:
                json.dump(self.model.__dict__, f)

        
