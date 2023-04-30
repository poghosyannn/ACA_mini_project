import argparse
import json
import pandas as pd
from Preproccesing import Preprocessing
from Model import Model
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, matthews_corrcoef



class Pipeline:
    def __init__(self):
        self.preprocessor = Preprocessing()
        self.model = Model()

    def run(self, data_path, inference):
        if inference == 'train':

            # Loading the data
            data = pd.read_csv(data_path)
            if 'recordid' in data:
                data = data.drop(['recordid'], axis=1)
            y = data['In-hospital_death']
            X = data.drop(columns=['In-hospital_death'])

            # Preprocessing
            self.preprocessor.fit(X, y)

            # Model training
            self.model.fit(self.preprocessor.balanced_Xtrain, self.preprocessor.balanced_Ytrain)

        elif inference == 'test':
            # Loading the data
            data = pd.read_csv(data_path)
            data = data.drop(['recordid'], axis=1)

            X_test = data.drop(columns=['In-hospital_death'])
            y_test = data['In-hospital_death']

            # Preprocessing
            X_test_processed = self.preprocessor.transform(X_test)

            # Make predictions
            y_pred = self.model.predict(X_test_processed)
            predict_probas = self.model.predict_proba(X_test_processed)
            predictions = {"predict_probas": predict_probas, "threshold": 0.4}

            with open('predictions.json', 'w') as f:
                json.dump(predictions, f)

            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)

            # Print the evaluation metrics
            print('Accuracy score:', accuracy)
            print('Recall score:', recall)
            print('AUC score:', auc)
            print('MCC score:', mcc)

            return y_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--inference', type=str, default='train', required=False)
    args = parser.parse_args()
    pipeline = Pipeline()
    pipeline.run(args.data_path, args.inference)
