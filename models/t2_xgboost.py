##
import numpy as np
import pandas as pd
from keras.losses import mean_absolute_error
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb
import copy
import matplotlib.pyplot as plt

from utils import path

##
pd.set_option('display.max_columns', None)
molecular_descriptor = pd.read_csv(path.molecular_descriptor_train_path)

molecular_descriptor = pd.DataFrame(molecular_descriptor,
                                    columns = ['BCUTc-1l', 'maxsOH', 'SHsOH', 'maxssO', 'MDEC-23',
                                               'LipoaffinityIndex', 'maxHsOH', 'minHsOH', 'minsOH', 'MLFER_A', 'VABC',
                                               'nHBAcc', 'MLogP', 'CrippenMR', 'ATSp1', 'nC', 'ATSp2', 'fragC', 'apol',
                                               'BCUTc-1h', ])

print(molecular_descriptor.head())

era_activity = pd.read_csv(path.era_activity_train_path)
era_activity = pd.DataFrame(era_activity, columns = ['pIC50'])

print(era_activity.head())

#
print(molecular_descriptor.values.shape)
print(era_activity.values.shape)

x_train = molecular_descriptor.values
y_train = era_activity.values

original_x = copy.copy(x_train)
original_y = copy.copy(y_train.reshape(len(y_train)))

x_scaler = preprocessing.MinMaxScaler()
x_train = x_scaler.fit_transform(x_train)
# print(x_train)

y_scaler = preprocessing.MinMaxScaler()
y_train = y_scaler.fit_transform(y_train)
y_train = y_train.reshape(len(y_train))
print(y_train)

data_matrix = xgb.DMatrix(x_train, y_train)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2, shuffle = False)
x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(original_x, original_y, test_size = 0.2, shuffle = False)

print(y_test[0:20])
print(y_test_o[0:20])

##
params = {"objective": "reg:linear", 'colsample_bytree': 0.3, 'learning_rate': 0.1,
          'max_depth': 5, 'alpha': 10}
cv_results = xgb.cv(dtrain = data_matrix, params = params, nfold = 3,
                    num_boost_round = 50, early_stopping_rounds = 10, metrics = "rmse", as_pandas = True, seed = 123)
cv_results.head()

##
xg_reg = xgb.train(params = params, dtrain = data_matrix, num_boost_round = 10)

xgb.plot_tree(xg_reg, num_trees = 0)
plt.rcParams['font.sans-serif'] = ['SF Mono']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 360  # ????????????
plt.rcParams['figure.dpi'] = 360  # ?????????
plt.rcParams['figure.figsize'] = [80, 60]
plt.show()

##
# xgb.plot_importance(xg_reg)
# plt.rcParams['figure.figsize'] = [3, 3]
# plt.show()

##
cv_params = {'n_estimators': [n for n in range(110, 130)]}
params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
          'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

model = xgb.XGBRegressor(**params)
optimized_GBM = GridSearchCV(estimator = model, param_grid = cv_params, scoring = 'r2', cv = 5, verbose = 1, n_jobs = 4)
optimized_GBM.fit(x_train, y_train)

print(optimized_GBM.best_params_)
print(optimized_GBM.best_score_)
##

cv_params = {'max_depth': [3, 4, 5, 6, 7], 'min_child_weight': [1, 2, 3]}
params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
          'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
model = xgb.XGBRegressor(**params)
optimized_GBM = GridSearchCV(estimator = model, param_grid = cv_params, scoring = 'r2', cv = 5, verbose = 1, n_jobs = 4)
optimized_GBM.fit(x_train, y_train)
print(optimized_GBM.best_params_)
print(optimized_GBM.best_score_)

##

cv_params = {'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 4, 'min_child_weight': 5, 'seed': 0,
          'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
model = xgb.XGBRegressor(**params)
optimized_GBM = GridSearchCV(estimator = model, param_grid = cv_params, scoring = 'r2', cv = 5, verbose = 1, n_jobs = 4)
optimized_GBM.fit(x_train, y_train)
print(optimized_GBM.best_params_)
print(optimized_GBM.best_score_)

##
cv_params = {'subsample': [n / 10 for n in range(5, 15)]}
params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 4, 'min_child_weight': 5, 'seed': 0,
          'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
model = xgb.XGBRegressor(**params)
optimized_GBM = GridSearchCV(estimator = model, param_grid = cv_params, scoring = 'r2', cv = 5, verbose = 1, n_jobs = 4)
optimized_GBM.fit(x_train, y_train)
print(optimized_GBM.best_params_)
print(optimized_GBM.best_score_)

##

cv_params = {'colsample_bytree': [n / 10 for n in range(1, 10)]}
params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 4, 'min_child_weight': 5, 'seed': 0,
          'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
model = xgb.XGBRegressor(**params)
optimized_GBM = GridSearchCV(estimator = model, param_grid = cv_params, scoring = 'r2', cv = 5, verbose = 1, n_jobs = 4)
optimized_GBM.fit(x_train, y_train)
print(optimized_GBM.best_params_)
print(optimized_GBM.best_score_)

##
cv_params = {'reg_alpha': [n / 10 for n in range(1, 10)]}
params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 4, 'min_child_weight': 5, 'seed': 0,
          'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
model = xgb.XGBRegressor(**params)
optimized_GBM = GridSearchCV(estimator = model, param_grid = cv_params, scoring = 'r2', cv = 5, verbose = 1, n_jobs = 4)
optimized_GBM.fit(x_train, y_train)
print(optimized_GBM.best_params_)
print(optimized_GBM.best_score_)

##

cv_params = {'reg_lambda': [n / 10 for n in range(1, 10)]}
params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 4, 'min_child_weight': 5, 'seed': 0,
          'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
model = xgb.XGBRegressor(**params)
optimized_GBM = GridSearchCV(estimator = model, param_grid = cv_params, scoring = 'r2', cv = 5, verbose = 1, n_jobs = 4)
optimized_GBM.fit(x_train, y_train)
print(optimized_GBM.best_params_)
print(optimized_GBM.best_score_)

##

cv_params = {'learning_rate': [n / 100 for n in range(1, 20, 2)]}
params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 4, 'min_child_weight': 5, 'seed': 0,
          'subsample': 0.7, 'colsample_bytree': 0.7, 'gamma': 0.1, 'reg_alpha': 1, 'reg_lambda': 1}

model = xgb.XGBRegressor(**params)
optimized_GBM = GridSearchCV(estimator = model, param_grid = cv_params, scoring = 'r2', cv = 5, verbose = 1, n_jobs = 4)
optimized_GBM.fit(x_train, y_train)
print(optimized_GBM.best_params_)
print(optimized_GBM.best_score_)

##
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, shuffle = False)
x_train_o, x_val_o, y_train_o, y_val_o = train_test_split(x_train_o, y_train_o, test_size = 0.2, shuffle = False)


xg_reg = xgb.XGBRegressor(
    n_estimators = 124,
    objective = 'reg:linear',
    max_depth = 4,
    min_child_weight = 3,
    gamma = 0.5,
    reg_alpha = 0.6,
    reg_lambda = 0.9,
    colsample_bytree = 0.5,
    learning_rate = 0.15,
    subsample=0.5
)
xg_reg.fit(x_train, y_train, eval_set = [(x_val, y_val)], verbose = True, early_stopping_rounds = 10)

##
pred = xg_reg.predict(x_train)
pred = y_scaler.inverse_transform(pred.reshape(len(pred), 1))
pred = pred.reshape(len(pred))
print(len(pred))
print(len(y_train_o))
print("Mean squared error: %.2f"
      % mean_squared_error(pred, y_train_o))
print('Variance score: %.2f' % r2_score(pred, y_train_o))

##
print(original_x.shape)
print(x_test.shape)
pred = xg_reg.predict(x_test)
pred = y_scaler.inverse_transform(pred.reshape(len(pred), 1))
pred = pred.reshape(len(pred))
print("Mean squared error: %.2f"
      % mean_squared_error(pred, y_test_o))
print('Variance score: %.2f' % r2_score(pred, y_test_o))
