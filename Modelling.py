# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 18:05:20 2021

@author: Dragox.RS
"""

# Imports
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, Ridge
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error,make_scorer,f1_score,precision_score,recall_score,accuracy_score
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt
import re

# Definitions
pd.set_option('display.float_format', lambda x: '%.3f' % x)
%matplotlib inline
#njobs = 4



# Get data
train = pd.read_csv("out.csv")

train = train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x)) #for_lightgbm_model_json_solution


X = train[:train.shape[0]]
del X['SalePrice']

y = train.SalePrice

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

#Models
# Linear_models

pip_reg = make_pipeline(RobustScaler(), Lasso())

pip_reg = Pipeline([
    ('robust', RobustScaler()),
    ('regr', Lasso())
])

param_grid = [
    {
        'regr': [Lasso(), Ridge(), ElasticNet()],
        'regr__alpha': [np.logspace(-4, 1, 8),15,10.5],
    }
]

grid = GridSearchCV(pip_reg, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0)
grid.fit(X_train, y_train)

predicted = grid.predict(X_test)


print(grid.best_params_)
print('Score of best regressor :\t{}'.format(grid.score(X_test, y_test)))


#RandomForest

rf = RandomForestRegressor(random_state=42)

# #Empirical good default values are max_features=(n_features)/3 for regression problems, and max_features=sqrt(n_features) for classification tasks
list_max_features=list(map(int,np.around(np.logspace(-4, 1, 8)+74)))
paramrf_grid = [
    {
      'n_estimators':[100,86,85,81,90],
    'max_features':list_max_features,
    'max_depth': [5,6,7,8]
    }
]

grid_rf=GridSearchCV(rf, param_grid=paramrf_grid, cv=6, n_jobs=-1, verbose=0)
grid_rf.fit(X_train, y_train)

prediction_rf = grid_rf.predict(X_test)

print(grid_rf.best_params_)
print('Score of best rf_regressor :\t{}'.format(grid_rf.score(X_test, y_test)))

#--- List of important features ---

features_list = X_train.columns.values
feature_importance = grid_rf.best_estimator_.feature_importances_
sorted_idx = np.argsort(feature_importance)

print(sorted_idx)

plt.figure(figsize=(15, 15))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature importances')
plt.draw()
plt.show()


#XGBoost

xgb_model = xgb.XGBClassifier(objective = "binary:logistic")

params = {
            'eta': np.arange(0.1, 0.26, 0.05),
            'min_child_weight': np.arange(1, 5, 0.5).tolist(),
            'gamma': [5],
            'subsample': np.arange(0.5, 1.0, 0.11).tolist(),
            'colsample_bytree': np.arange(0.5, 1.0, 0.11).tolist()
        }

scorers = {
            'f1_score':make_scorer(f1_score),
            'precision_score': make_scorer(precision_score),
            'recall_score': make_scorer(recall_score),
            'accuracy_score': make_scorer(accuracy_score)
          }



xgb_reg = xgb.XGBRegressor()

parameters = {'objective':['reg:linear'],
              'learning_rate': [0.03], #so called `eta` value
              'max_depth': [5],
              'min_child_weight': [4],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500]}

grid_xgb = GridSearchCV(xgb_reg, 
                    param_grid = parameters, 
                    n_jobs = -1, 
                    cv = 7,
                    refit = "accuracy_score",
                    verbose=0)

grid_xgb.fit(X_train, y_train)

prediction_xgb = grid_xgb.predict(X_test)

print(grid_xgb.best_params_)
print('Score of best xgb_regressor :\t{}'.format(grid_xgb.score(X_test, y_test)))


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11,silent=True)

model_lgb.fit(X_train, y_train)
prediction_lgb = model_lgb.predict(X_test)


print('Score of best lgb_regressr :\t{}'.format(model_lgb.score(X_test, y_test)))

# #Results

# {'regr': Ridge(alpha=15), 'regr__alpha': 15}
# Score of best regressor :	0.9113107783963508

# {'max_depth': 8, 'max_features': 84, 'n_estimators': 100}
# Score of best rf_regressor :	0.849346432226536


=# {'colsample_bytree': 0.7, 'learning_rate': 0.03, 'max_depth': 5, 'min_child_weight': 4, 'n_estimators': 500, 'objective': 'reg:linear', 'subsample': 0.7}
# Score of best xgb_regressor :	0.897131087620481

# Score of best lgb_regressr :	0.897780825244955