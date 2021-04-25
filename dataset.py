import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self,data_file): 
        """
        Parameters
        data_file : (string)CSV file location
        """
        self.dataset = pd.read_csv(data_file)  #dataset is a dataframe

    def scale_data(self,columns_to_scale):
        """
        Parameters
        columns_to_scale : (list)columns names to be scaled
        """
        standardScaler = MinMaxScaler
        #columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        self.dataset[columns_to_scale] = standardScaler().fit_transform(self.dataset[columns_to_scale])

    def spilt_data(self,y_column_name, test_size=0.2):
        """
        Parameters
        y_column_name : (string) column to be taken as target
        """
        y = self.dataset[y_column_name]
        X = self.dataset.drop([y_column_name], axis = 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 0)
        return X_train, X_test, y_train, y_test

    def get_cols(self):
        return list(self.dataset.columns)