# Importing needed libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score,confusion_matrix, matthews_corrcoef, f1_score
from imblearn.over_sampling import RandomOverSampler,SMOTE


# Reading csv file, doing train test split
df = pd.read_csv('hospital_deaths_train.csv')
df = df.drop('recordid', axis=1)
X = df.drop('In-hospital_death', axis=1)
y = df['In-hospital_death']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# after spliting the data reseting indexes
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
df_train = pd.concat([y_train, X_train], axis=1)

X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
df_test = pd.concat([y_test, X_test], axis=1)


# Simple imputer with indicator columns being added
copy_train = X_train.copy()
simple_imputer = SimpleImputer(add_indicator=True)
X_train = simple_imputer.fit_transform(X_train)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

sm = RandomOverSampler(random_state=42)
balanced_Xtrain, balanced_Ytrain = sm.fit_resample(X_train_scaled, y_train)


X_test = simple_imputer.transform(X_test)

X_test = scaler.transform(X_test)


svmrbf = SVC(kernel = 'rbf', C=6, gamma=0.02, probability=True)
rndfr = RandomForestClassifier(n_estimators=300)
logr = LogisticRegression(C=1, penalty = 'l2')
qda = QuadraticDiscriminantAnalysis()
ada = AdaBoostClassifier(n_estimators=230)

voting = VotingClassifier(estimators= [('svmrbf', svmrbf), ('rndfr', rndfr), ('logr', logr), 
                                       ('qda', qda), ('ada', ada)], voting = 'soft').fit(balanced_Xtrain, balanced_Ytrain)



y_pred = voting.predict(X_test)
y_pred_proba = voting.predict_proba(X_test)

threshold = 0.4
y_pred = (voting.predict_proba(X_test)[:, 1] >= threshold).astype(int)


accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
specificity = precision_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

# Print the evaluation metrics
print('Accuracy score:', accuracy)
print('Recall score:', recall)
print('Specificity score:', specificity)
print('AUC score:', auc)
print('MCC score:', mcc)
