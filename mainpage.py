# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 02:19:10 2019

@author: meet
"""

from tkinter import filedialog
from tkinter import *

def show_result():
    print(master.filename)
    model=variable.get()
    x_selected=x_axis.get()
    y_selected=y_axis.get()
    x_selected_index=x_axis_int.get()
    y_selected_index=y_axis_int.get()    
    print(model)
    print(x_selected)
    print(y_selected)
    print(x_selected_index)
    print(y_selected_index)
    if(model=="Linear Regression"):
        import pandas as pd
        data=pd.read_csv(master.filename)
        import matplotlib.pyplot as plt
        array = data.values
        X = array[:,x_selected_index:x_selected_index+1]
        Y = array[:,y_selected_index]
        from sklearn.model_selection import train_test_split  
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, shuffle= True)    
        from sklearn.linear_model import LinearRegression  
        regressor = LinearRegression()  
        regressor.fit(X_train,Y_train)
        Y_pred = regressor.predict(X_test)
        type(Y_pred)
        X_test
        df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})  
        a=df["Actual"]
        pd.to_numeric(a)
        b=df["Predicted"]
        pd.to_numeric(b)
        plt.scatter(X_test,Y_test,color='red')
        plt.plot(X_test,regressor.predict(X_test),color='blue')
        plt.grid(True)
        plt.show()    
    
    
    
    
def open_file():    
    master.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    print (master.filename)
    path=master.filename
    Label(master,text=master.filename).grid(column=1,row=2)

OPTIONS = [
"Select model",
"Linear Regression",
"Logistic Regression",
"Polynomial Regression",
"Stepwise Regression"
] #etc

master = Tk()
master.geometry("400x400")
variable = StringVar(master)
variable.set(OPTIONS[0]) # default value

w=OptionMenu(master, variable, *OPTIONS)
w.grid(column=0,row=1)

b2=Button(master,text="select CSV File",command=open_file)
b2.grid(column=0,row=2)

#x_axis=IntVar()
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


b2=Button(master,text="show result",command=show_result)
b2.grid(column=1,row=7)


mainloop()