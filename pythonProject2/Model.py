# Importing needed libs
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import Preproccesing
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, confusion_matrix, roc_auc_score, f1_score, matthews_corrcoef


class Model:
    def __init__(self):
        self.logr = None

    def fit(self, X_train, y_train):
        self.logr = LogisticRegression(C=15, penalty='l2').fit(X_train, y_train)

    def predict(self, X_test):
        threshold = 0.5
        y_pred = self.logr.predict(X_test)
        y_pred = (self.logr.predict_proba(X_test)[:, 1] >= threshold).astype(int)

        return y_pred

    def predict_proba(self, X_test):

        return self.logr.predict_proba(X_test)


df = pd.read_csv('hospital_deaths_train.csv')
df = df.drop('recordid', axis=1)
X = df.drop('In-hospital_death', axis=1)
y = df['In-hospital_death']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


preprocessor = Preproccesing.Preprocessing()
preprocessor.fit(X_train, y_train)
X_test = preprocessor.transform(X_test)
X_train = preprocessor.balanced_Xtrain
y_train = preprocessor.balanced_Ytrain

model = Model()
model.fit(X_train, y_train)
Y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)

print(model.predict(X_test))
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Recall', recall_score(y_test, y_pred))
print('precision', precision_score(y_test, y_pred))
print('auc', roc_auc_score(y_test, y_pred))
print('f1_score', f1_score(y_test, y_pred))
print('MCC', matthews_corrcoef(y_test, y_pred))
