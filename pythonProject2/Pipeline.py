class My_Pipeline:
    def __init__(self):
        self.preprocessor = Preprocessing()
        self.model = Model()
        
    def run(self, data_path, mode):
        if mode == 'train':
            
            #Loading the data 
            data = pd.read_csv(data_path)
            data = data.drop(['recordid'], axis = 1)
            X = data.drop(columns=['In-hospital_death'])
            y = data['In-hospital_death']
            
            # Preprocessing
            self.preprocessor.fit(X, y)
            
            # Model training
            self.model.fit(self.preprocessor.balanced_Xtrain, self.preprocessor.balanced_Ytrain)
            
            
        elif mode == 'test':
            #Loading the data 
            data = pd.read_csv(data_path)
            data = data.drop(['recordid'], axis = 1)

            X_test = data.drop(columns=['In-hospital_death'])
            y_test = data['In-hospital_death']            
            
            # Preprocessing
            X_test_processed = self.preprocessor.transform(X_test)
            
            # Make predictions
            y_pred = self.model.predict(X_test_processed)
            
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
