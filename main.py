from sklearn.metrics import accuracy_score,confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from classification import Classification
from regression import Regression
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics
from dataset import Dataset
import numpy as np


# Get Dataset
iris = datasets.load_boston()

# Get x and y
x = iris.data
y = iris.target

Classifier_list=["RandomForestClassifier","NaiveBayes","SupportVectorMachine","StochasticGradientDescent","KNN"]
Regressor_list=['RandomForestRegressor','DecissionTreeRegressor','ADABoostRegressor','RidgeRegressor','SGDRegressor',
'SupportVectorRegression','ARDRegression']

######Manipulating the data

filename = input("Enter CSV file location: ")
data = Dataset(filename)

print("These are the columns you have: "+str(data.get_cols()))

print("Enter column names to scale")
print("enter Y when you are done inputting names")
cols=" "
cols_list=[]
while cols!="Y":
    cols=input()
    cols_list.append(cols)
data.scale_data(cols_list[:-1])


cols_list = data.get_cols()
y_column_name=input("enter y column name: ")
X_train, X_test, y_train, y_test = data.spilt_data(y_column_name)

model_type = input("Enter R for Regression and C for Classification: ")


if model_type == "C":
    print("Your options are: "+str(Classifier_list))#add mode list
    modelname = input("Enter model to be used: ")
    classifier =  Classification(X_train, X_test, y_train, y_test, modelname)
    classifier.predict()
    classifier.accuracy()
    classifier.save_model()

elif model_type=='R':
    print("Your options are: "+str(Regressor_list))#add mode list
    modelname = input("Enter model to be used, use A for all")
    if modelname == "A":
        for modelname in Regressor_list:
            regressor = Regression(X_train, X_test, y_train, y_test, modelname)
            regressor.predict()
            regressor.accuracy()
            regressor.save_model()
    else:
        regressor = Regression(X_train, X_test, y_train, y_test, modelname)
        regressor.predict()
        regressor.accuracy()
        regressor.save_model()