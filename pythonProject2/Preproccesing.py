# Importing needed libs
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split


class Preprocessing:
    def __init__(self):
        self.balanced_Xtrain = None
        self.balanced_Ytrain = None
        self.simple_imputer = None
        self.scaler = None
        pass

    def fit(self, X_train, y_train):
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        df_train = pd.concat([y_train, X_train], axis=1)

        # Simple imputer with indicator columns being added
        copy_train = X_train.copy()
        self.simple_imputer = SimpleImputer(add_indicator=True)
        X_train = self.simple_imputer.fit_transform(X_train)

        original_column_names = list(copy_train.columns)
        transformed_column_names = np.append(original_column_names, self.simple_imputer.indicator_.features_)

        X_train = pd.DataFrame(X_train, columns=transformed_column_names)

        self.scaler = MinMaxScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        sm = RandomOverSampler(random_state=42)
        self.balanced_Xtrain, self.balanced_Ytrain = sm.fit_resample(X_train_scaled, y_train)

    def transform(self, X_test):
        X_test = X_test.reset_index(drop=True)

        test_copy = X_test.copy()

        X_test = self.simple_imputer.transform(X_test)
        original_column_names = list(test_copy.columns)
        transformed_column_names = np.append(original_column_names, self.simple_imputer.indicator_.features_)
        X_test = pd.DataFrame(X_test, columns=transformed_column_names)

        X_test = self.scaler.transform(X_test)
        X_test = pd.DataFrame(X_test, columns=transformed_column_names)

        return X_test


df = pd.read_csv('hospital_deaths_train.csv')
df = df.drop('recordid', axis=1)
X = df.drop('In-hospital_death', axis=1)
y = df['In-hospital_death']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = Preprocessing()

# Fit and transform the training set
preprocessor.fit(X_train, y_train)

# Transform the training and testing sets
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(X_test_processed)

