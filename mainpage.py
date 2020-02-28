# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 02:19:10 2019

@author: meet
"""

from tkinter import filedialog
from tkinter import *
import pandas as pd
dataset = pd.read_csv('dataset.csv')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.cm import rainbow
import warnings
warnings.filterwarnings('ignore')
rcParams['figure.figsize'] = 10, 10
from sklearn.preprocessing import MinMaxScaler
standardScaler = MinMaxScaler
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler().fit_transform(dataset[columns_to_scale])
# create a figure and axis 
from sklearn.model_selection import train_test_split
y = dataset['target']
X = dataset.drop(['target'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



def show_result():
    fp=master.filename
    print(fp)
    model=variable.get()

    if (model=='Naive Bayes Classifier'):
        print('code for Naive Bayes Classifier')
        #importing naive bayes and accuracy mattrix to measure model performance
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import accuracy_score
        data1=pd.read_csv(fp)
        standardScaler = MinMaxScaler
        columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        data1[columns_to_scale] = standardScaler().fit_transform(data1[columns_to_scale])
        X_test1 = data1[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
        y_test1 = data1[['target']]
        
        model = GaussianNB()
        # fit the model with the training data
        model.fit(X_train,y_train)
        predict_train = model.predict(X_train)
        print('Target on train data',predict_train)
        # predict the target on the train dataset
        accuracy_train = accuracy_score(y_train,predict_train)
        print('accuracy_score on train dataset : ', accuracy_train*100)
        # predict the target on the test dataset
        predict_test = model.predict(X_test1)
        print('Target on test data',predict_test)
        accuracy_test = accuracy_score(y_test1,predict_test)
        print('accuracy_score on test dataset : ', accuracy_test*100)
    elif(model=='KNN'):
        #below is the code for the knn classifier with neighbors ranging from 1-4
        from sklearn.neighbors import KNeighborsClassifier    
        from sklearn.metrics import accuracy_score
        data1=pd.read_csv(fp)
        standardScaler = MinMaxScaler
        columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        data1[columns_to_scale] = standardScaler().fit_transform(data1[columns_to_scale])
        X_test1 = data1[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
        y_test1 = data1[['target']]                
        knn_scores = []
        for k in range(1,5):
            knn_classifier = KNeighborsClassifier(n_neighbors = k)
            knn_classifier.fit(X_train, y_train)
            knn_scores.append(knn_classifier.score(X_test, y_test))
        knn_scores
        print("We got highest accuracy of {}% with {} nieghbors in KNN clasifier".format(knn_scores[2]*100, 3))
    elif(model=='Random Forest'):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        data1=pd.read_csv(fp)
        standardScaler = MinMaxScaler
        columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        data1[columns_to_scale] = standardScaler().fit_transform(data1[columns_to_scale])
        X_test1 = data1[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
        y_test1 = data1[['target']]  
        randomForest = RandomForestClassifier()
        randomForest.fit(X_train,y_train)
        predict_train = randomForest.predict(X_train)
        print('Target on train data',predict_train)
        # predict the target on the train dataset
        accuracy_train = accuracy_score(y_train,predict_train)
        print('accuracy_score on train dataset : ', accuracy_train*100)
        # predict the target on the test dataset
        predict_test = randomForest.predict(X_test1)
        print('Target on test data',predict_test)
        accuracy_test = accuracy_score(y_test1,predict_test)
        print('accuracy_score on test dataset : ', accuracy_test*100)
    elif(model=='Decision Tree'):
        from sklearn import tree
        from sklearn.metrics import accuracy_score
        data1=pd.read_csv(fp)
        standardScaler = MinMaxScaler
        columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        data1[columns_to_scale] = standardScaler().fit_transform(data1[columns_to_scale])
        X_test1 = data1[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
        y_test1 = data1[['target']] 
        decisionTree=tree.DecisionTreeClassifier()
        decisionTree.fit(X_train,y_train)
        predict_train = decisionTree.predict(X_train)
        print('Target on train data',predict_train)
        # predict the target on the train dataset
        accuracy_train = accuracy_score(y_train,predict_train)
        print('accuracy_score on train dataset : ', accuracy_train*100)
        # predict the target on the test dataset
        predict_test = decisionTree.predict(X_test1)
        print('Target on test data',predict_test)
        accuracy_test = accuracy_score(y_test1,predict_test)
        print('accuracy_score on test dataset : ', accuracy_test*100)
    
def open_file():    
    master.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    print (master.filename)
    path=master.filename
    Label(master,text=master.filename).grid(column=1,row=2)

OPTIONS = [
"Select model",
"KNN",
"Naive Bayes Classifier",
"Random Forest",
"Decision Tree"
] #etc

master = Tk()
master.geometry("400x400")
variable = StringVar(master)
variable.set(OPTIONS[0]) # default value

w=OptionMenu(master, variable, *OPTIONS)
w.grid(column=0,row=1)

b2=Button(master,text="select CSV File",command=open_file)
b2.grid(column=0,row=2)

'''#x_axis=IntVar()
Label(master,text="x_axis").grid(column=0,row=3)
x_axis=StringVar()
e1=Entry(master,width=20,textvariable=x_axis)
e1.grid(column=0,row=4)
    
#y_axis=IntVar()
Label(master,text="y_axis").grid(column=1,row=3)
y_axis=StringVar()
e2=Entry(master,width=20,textvariable=y_axis)
e2.grid(column=1,row=4)


x_axis_int=IntVar()
Label(master,text="array index of x").grid(column=0,row=5)
e1=Entry(master,width=20,textvariable=x_axis_int)
e1.grid(column=0,row=6)
    
y_axis_int=IntVar()
Label(master,text="array index of y").grid(column=1,row=5)
e2=Entry(master,width=20,textvariable=y_axis_int)
e2.grid(column=1,row=6)
'''

b2=Button(master,text="show result",command=show_result)
b2.grid(column=3,row=1)


mainloop()
