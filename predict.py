import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
#from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.linear_model import LogisticRegression,SGDClassifier
#from sklearn.model_selection import train_test_split, GridSearchCV
import re
import numpy as np

def get_title(name):
    title = re.search(' ([A-Za-z]+)\.',name)
    if title:
        if title.group(1) in ['Mr','Mrs','Miss','Master']:
            return title.group(1)
        else:
            return 'Other'
    else:
        return ''

#open the datasets
df_train = pd.read_csv('train.csv')
df_pred = pd.read_csv('test.csv')

#create new features

#encode sex as integer
df_train['male'] = pd.get_dummies(df_train.Sex,drop_first=True)
df_pred['male'] = pd.get_dummies(df_pred.Sex,drop_first=True)

#get family size as siblings + parents
df_train['FamilySize'] = df_train.Parch+df_train.SibSp
df_pred['FamilySize'] = df_pred.Parch+df_pred.SibSp

#extract title / Salutation
df_train['Salutation'] = df_train.Name.apply(get_title)
df_pred['Salutation'] = df_pred.Name.apply(get_title)

#change class to categorical values
df_train.Pclass = df_train.Pclass.astype('category')
df_pred.Pclass = df_pred.Pclass.astype('category')

#define predictors
predictors = ['Pclass','SibSp','Parch','male','FamilySize','Salutation']

#create dummies for Salutation
X = pd.get_dummies(df_train[predictors],drop_first=True)
y = df_train.Survived

#create train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=42)

#set up the pipeline
steps = [('scaler',StandardScaler()), ('clf',SVC())]

pipeline = Pipeline(steps)

parameters = {'clf__C': 10**np.arange(-4.,4.),
              'clf__gamma': 10**np.arange(-5.,0)}

grid_cv = GridSearchCV(pipeline,param_grid=parameters,cv=5)
grid_cv.fit(X_train,y_train)

print('Results from GridSearchCV')
print('Best pars:', grid_cv.best_params_)
print('Best score:', grid_cv.best_score_)
print ('Score for out-of-sample:', grid_cv.score(X_test,y_test))

#now fit on full dataset with best parameters
#grid_cv.best_estimator_.fit(X,y)
print('Score on whole dataset:', grid_cv.score(X,y))

#get X for predictions as X_pred
X_pred = pd.get_dummies(df_pred[predictors],drop_first=True)

pred=pd.DataFrame()
pred['Survived'] = grid_cv.predict(X_pred)
pred.index = df_pred.PassengerId
pred.to_csv('submission_18072018.csv')
