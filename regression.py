from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
import pickle

class Regression:
    def __init__(self,X_train, X_test, y_train, y_test, Model):
        """[summary]

        Args:
            X_train ([dataframe])
            X_test ([dataframe])
            y_train ([dataframe])
            y_test ([dataframe])
            modelname ([string])
        """
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        self.Model=Model

    def accuracy(self):
        """ Show accuracy of prediction with trained model.

        Args:
            y_pred ([list]): [list of predicted value from x_test]
            y_test ([list]): [true value of x_test]
            Model_Type ([string]): [is it a classification or regression problem.]
        """

        predict_train = self.trained_model.predict(self.X_train)
        predict_test = self.trained_model.predict(self.X_test)
        print("Train MAE: ",metrics.mean_absolute_error(self.y_test,predict_train))
        print("Train RMSE: ",np.sqrt(metrics.mean_squared_error(self.y_test,predict_train)))   

        print("Test MAE: ",metrics.mean_absolute_error(self.y_test,predict_test))
        print("Test RMSE: ",np.sqrt(metrics.mean_squared_error(self.y_test,predict_test)))   

    def save_model(self):
        f = open(self.Model+'.pickle', 'wb')
        pickle.dump(self.trained_model, f)
        f.close()

    def predict(self):
        if Model=="Random Forest Regressor":
            self.self.trained_model=RandomForestRegressor(n_estimators=200,min_samples_split=2,min_samples_leaf=2,max_features='sqrt', max_depth=80, bootstrap=True)
            self.trained_model.fit(self.X_train,self.y_train)
        elif Model=="Decission Tree Regressor":
            self.trained_model=DecisionTreeRegressor(max_depth=4)
            self.trained_model.fit(self.X_train,self.y_train)
        elif Model=="ADA Boost Regressor":
            self.trained_model=AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=291)
            self.trained_model.fit(self.X_train,self.y_train)
        elif Model=="Ridge Regressor":
            self.trained_model=Ridge(alpha=50)
            self.trained_model.fit(self.X_train,self.y_train)
        elif Model=="SGD Regressor":
            self.trained_model=SGDRegressor(max_iter=1000,tol=1e-3)
            self.trained_model.fit(self.X_train,self.y_train)
        elif Model=="Support Vector Regression":
            self.trained_model = svm.SVR(kernel='rbf')
            self.trained_model.fit(self.X_train,self.y_train)
        elif Model=="ARDRegression":
            self.trained_model=ARDRegression() 
            self.trained_model.fit(self.X_train,self.y_train)
