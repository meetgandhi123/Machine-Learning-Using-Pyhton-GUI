from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import svm
import pickle

class Classification:
    def __init__(self,X_train, X_test, y_train, y_test, modelname):
        """[summary]

        Args:
            X_train ([dataframe])
            X_test ([dataframe])
            y_train ([dataframe])
            y_test ([dataframe])
            modelname ([string])
        """
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        self.modelname=modelname

    def accuracy(self):
        predict_train = self.trained_model.predict(self.X_train)
        accuracy_train = accuracy_score(self.y_train,predict_train)
        predict_test = self.trained_model.predict(self.X_test)
        accuracy_test = accuracy_score(self.y_test,predict_test)
        print(self.modelname+": Training accuracy: "+ str(accuracy_train))
        print(self.modelname+": Testing  accuracy: "+ str(accuracy_test))

    def save_model(self):
        f = open(self.modelname+'.pickle', 'wb')
        pickle.dump(self.trained_model, f)
        f.close()

    def predict(self):
        if self.modelname =='RandomForestClassifier':
            self.trained_model = RandomForestClassifier(n_estimators=100, bootstrap = True, max_features = 'sqrt')
            self.trained_model.fit(self.X_train,self.y_train)
        elif self.modelname =="NaiveBayes":
            gnb = GaussianNB()
            self.trained_model = gnb.fit(self.X_train,self.y_train)
        elif self.modelname =="SupportVectorMachine":
            self.trained_model = svm.SVC()
            self.trained_model.fit(self.X_train,self.y_train)
        elif self.modelname =="StochasticGradientDescent":
            self.trained_model = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
            self.trained_model.fit(self.X_train,self.y_train)
            SGDClassifier(max_iter=5)
        elif self.modelname =='KNN':
            self.trained_model = KNeighborsClassifier(n_neighbors=2)
            self.trained_model.fit(self.X_train,self.y_train)