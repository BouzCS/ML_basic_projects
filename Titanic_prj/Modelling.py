# -*- coding: utf-8 -*-
"""
@author: Dragox.RS
"""
import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                             ExtraTreesClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score

# Definitions
pd.set_option('display.float_format', lambda x: '%.3f' % x)

#Loading Dataset

train = pd.read_csv("trainF.csv")
del train['Unnamed: 0']
train = train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x)) #for_lightgbm_model_json_solution


X = train[:train.shape[0]]
del X['Survived']
y = train.Survived

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

#Models
#Classifications

#Logistic_regression

model_lgr = LogisticRegression()

pip_lgr = Pipeline([
    ('standardscaler',StandardScaler()),
    ('pca',PCA()), 
    ('regr',LogisticRegression(solver='liblinear'))
])

param_lgr_grid = [
    {
        'pca__n_components':list(range(1,X.shape[1]+1,1)),
        'regr__C':np.logspace(-4, 4, 50),
        'regr__penalty':['l1', 'l2']
    }
]

grid_lgr = GridSearchCV(pip_lgr, param_grid=param_lgr_grid, cv=5, n_jobs=-1, verbose=2)
grid_lgr.fit(X_train, y_train)

predicted = grid_lgr.predict(X_test)


print(grid_lgr.best_params_)
print('Score of best classifier :\t{}'.format(grid_lgr.score(X_test, y_test)))

print(classification_report(y_test, predicted))

# #Decisive Tree Models

rf = RandomForestClassifier(random_state=12)

#Empirical good default values are max_features=(n_features)/3 for regression problems, and max_features=sqrt(n_features) for classification tasks

paramrf_grid = [
    {
      'n_estimators':[100,81,1200, 1400, 1600, 1800, 2000],
    'max_features':['sqrt'],
    'max_depth': [5,6,30],
    'min_samples_split': [10,20],
    'min_samples_leaf': [4,12]
    }
]

rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}

paramrf_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}



grid_rf=GridSearchCV(rf, param_grid=rf_param_grid, cv=6, n_jobs=-1, verbose=2)
grid_rf.fit(X_train, y_train)



prediction_rf = grid_rf.predict(X_test)

print(grid_rf.best_params_)
print('Score of best rf_classifier :\t{}'.format(grid_rf.score(X_test, y_test)))
print(classification_report(y_test, prediction_rf))

# {'bootstrap': False, 'criterion': 'gini', 'max_depth': None, 'max_features': 3, 'min_samples_leaf': 10, 'min_samples_split': 2, 'n_estimators': 300}
# Score of best rf_classifier :	0.823728813559322

#SVC
SVMC = SVC(probability=True)
    
svc_param_grid = {'kernel': ['rbf', 'poly', 'sigmoid','linear'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [0.1,1, 10, 50, 100, 1000]}

grid_svc=GridSearchCV(SVMC, param_grid=svc_param_grid, cv=6, n_jobs=-1, verbose=2)
grid_svc.fit(X_train, y_train)



prediction_svc = grid_svc.predict(X_test)

print(grid_svc.best_params_)
print('Score of best svc_classifier :\t{}'.format(grid_svc.score(X_test, y_test)))
print(classification_report(y_test, prediction_svc))


# {'C': 50, 'gamma': 0.01, 'kernel': 'rbf'}
# Score of best svc_classifier :	0.823728813559322
#               precision    recall  f1-score   support

#          0.0       0.84      0.86      0.85       174
#          1.0       0.79      0.77      0.78       121

#     accuracy                           0.82       295
#    macro avg       0.82      0.82      0.82       295
# weighted avg       0.82      0.82      0.82       295

# [Parallel(n_jobs=-1)]: Done 576 out of 576 | elapsed: 19.5min finished


#ADA_BOOST

TC = DecisionTreeClassifier(random_state=45)

ABC = AdaBoostClassifier(base_estimator = TC)


grid_ada_param={
            'base_estimator__criterion':['gini','entropy'],
            'base_estimator__splitter':['best','random'],
            'base_estimator__max_depth':[5,6,8],
            'base_estimator__min_samples_leaf':[6,10,12],
            'base_estimator__max_features':['sqrt','log2'],
            
            'n_estimators': [100,300,500],
            'learning_rate':[0.001,0.05,0.01,0.1],
            'algorithm':['SAMME', 'SAMME.R']
    }
grid_ada_param={
            'base_estimator__criterion':['gini'],
            'base_estimator__splitter':['best'],
            'base_estimator__max_depth':[3],
            'base_estimator__min_samples_leaf':[13,12],
            'base_estimator__max_features':['sqrt'],
            
            'n_estimators': [100],
            'learning_rate':[0.09,0.095],
            'algorithm':['SAMME']
    }

grid_ada=GridSearchCV(ABC, param_grid=grid_ada_param, cv=6, n_jobs=-1, verbose=2)
grid_ada.fit(X_train, y_train)



prediction_ada = grid_ada.predict(X_test)

print(grid_ada.best_params_)
print('Score of best ada_classifier :\t{}'.format(grid_ada.score(X_test, y_test)))
print(classification_report(y_test, prediction_ada))
# Score of best ada_classifier :	0.8338983050847457


# #XGBoost

xgb_model = xgb.XGBClassifier(objective = "reg:logistic")

params = {
            'eta': np.arange(0.1, 0.26, 0.05).tolist(),
            'min_child_weight': np.logspace(-2, 1.5, 3).tolist(),
            'gamma': [5,10,2],
            'subsample': np.logspace(-2, 1.5, 3),
            'colsample_bytree': np.logspace(-2, 1.5, 3).tolist(),
              'n_estimators': [100,300]
        }


xgb_clf = xgb.XGBClassifier(use_label_encoder=False)

parameters = {'objective':['binary:logitraw','binary:logistic','binary:hinge'],
              'learning_rate': [0.03], #so called `eta` value
              'max_depth': [5],
              'min_child_weight': [4],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500]}

grid_xgb = GridSearchCV(xgb_clf, 
                    param_grid = params, 
                    n_jobs = -1, 
                    cv = 5,
                    refit = "accuracy_score",
                    verbose=2)

grid_xgb.fit(X_train, y_train)

prediction_xgb = grid_xgb.predict(X_test)

print(grid_xgb.best_params_)
print('Score of best xgb_regressor :\t{}'.format(grid_xgb.score(X_test, y_test)))

# {'colsample_bytree': 0.01, 'eta': 0.20000000000000004, 'gamma': 2, 'min_child_weight': 0.01, 'n_estimators': 300, 'subsample': 0.5623413251903491}
# Score of best xgb_regressor :	0.8203389830508474


# Param_GRID
# grid_n_estimator = [10, 50, 100, 300]
# grid_ratio_max_samples = [.1, .25, .5, .75, 1.0]
# grid_learn = [.01, .03, .05, .1, .25]
# grid_max_depth = [2, 4, 6, 8, 10, None]
# grid_min_samples = [5, 10, .03, .05, .10]
# grid_criterion = ['gini', 'entropy']
# grid_bool = [True, False]
# grid_seed = [0]

#KNN

knn_model = KNeighborsClassifier()

knn_pipe = Pipeline([
        ('sc', StandardScaler()),     
        ('knn', KNeighborsClassifier()) 
    ])

knn_param = {
        'knn__n_neighbors': [7, 9, 13,11,15,19,21], # usually odd numbers
        'knn__weights': ['uniform', 'distance'],
        'knn__algorithm':['auto','ball_tree','kd_tree','brute'],
        'knn__leaf_size':np.arange(10,53,6).tolist(),
        'knn__p':[1,2]
    }

grid_knn = GridSearchCV(knn_pipe, 
                    param_grid = knn_param, 
                    n_jobs = -1, 
                    cv = 5,
                    refit = "accuracy_score",
                    verbose=2)

grid_knn.fit(X_train, y_train)

prediction_knn = grid_knn.predict(X_test)

print(grid_knn.best_params_)
print('Score of best knn_regressor :\t{}'.format(grid_knn.score(X_test, y_test)))


# {'knn__algorithm': 'auto', 'knn__leaf_size': 10, 'knn__n_neighbors': 7, 'knn__p': 2, 'knn__weights': 'uniform'}
# Score of best knn_regressor :	0.8372881355932204

