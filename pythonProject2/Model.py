# Importing needed libs
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import Preproccesing
from sklearn.model_selection import train_test_split


class Model:
    def __init__(self):
        self.voting = None
        self.predict_proba = None
        # self.predict = None

    def fit(self, X_train, y_train):
        svmrbf = SVC(kernel='rbf', C=6, gamma=0.02, probability=True)
        rndfr = RandomForestClassifier(n_estimators=300)
        logr = LogisticRegression(C=1, penalty='l2')
        qda = QuadraticDiscriminantAnalysis()
        ada = AdaBoostClassifier(n_estimators=230)

        self.voting = VotingClassifier(estimators=[('svmrbf', svmrbf), ('rndfr', rndfr), ('logr', logr),
                                              ('qda', qda), ('ada', ada)], voting='soft').fit(X_train, y_train)

    def predict(self, X_test):
        # threshold = 0.4
        predict = self.voting.predict(X_test)
        # y_pred = (self.voting.predict_proba(X_test)[:, 1] >= threshold).astype(int)
        return predict

    # def predict_proba(self, X_test):
    #     return self.voting.predict_proba(X_test)


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
predictions = model.predict(X_test)
print(predictions)




