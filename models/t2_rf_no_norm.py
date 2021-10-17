##
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import joblib

from utils import path

##
# 预处理
pd.set_option('display.max_columns', None)
molecular_descriptor = pd.read_csv(path.molecular_descriptor_train_path)

molecular_descriptor = pd.DataFrame(molecular_descriptor,
                                    columns = ['SHsOH', 'SsOH', 'LipoaffinityIndex', 'C2SP2', 'BCUTc-1l',
                                               'apol', 'minsOH', 'maxHsOH', 'MDEC-23', 'maxsssN', 'SwHBa',
                                               'MLFER_A', 'ATSp4', 'SP-5', 'nHsOH', 'maxsOH', 'nC', 'nHBAcc', 'nAtom',
                                               'minHsOH'])

print(molecular_descriptor.head())

era_activity = pd.read_csv(path.era_activity_train_path)
era_activity = pd.DataFrame(era_activity, columns = ['pIC50'])

print(era_activity.head())

#
print(molecular_descriptor.values.shape)
print(era_activity.values.shape)

x_train = molecular_descriptor.values
y_train = era_activity.values

# x_scaler = preprocessing.MinMaxScaler()
# x_train = x_scaler.fit_transform(x_train)
print(x_train)

# y_scaler = preprocessing.MinMaxScaler()
# y_train = y_scaler.fit_transform(y_train)
y_train = y_train.reshape(len(y_train))
print(y_train)

##
# 调n_estimators
plt.rcParams['font.sans-serif'] = ['SF Mono']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 360  # 图片像素
plt.rcParams['figure.dpi'] = 360  # 分辨率

param_grid = {'n_estimators': np.arange(100, 120, 1)}
rfc = RandomForestRegressor(n_estimators = 121,
                            random_state = 90)
GS = GridSearchCV(rfc, param_grid, cv = 3, verbose = 1)
GS.fit(x_train, y_train)

print(GS.best_params_)
print(GS.best_score_)

##
param_grid = {'random_state': np.arange(80, 100, 1)}
rfc = RandomForestRegressor(n_estimators = 121,
                            random_state = 90)
GS = GridSearchCV(rfc, param_grid, cv = 3, verbose = 1)
GS.fit(x_train, y_train)

print(GS.best_params_)
print(GS.best_score_)

##
# 调参
param_grid = {'max_depth': np.arange(20, 40, 1)}
#   一般根据数据的大小来进行一个试探，乳腺癌数据很小，所以可以采用1~10，或者1~20这样的试探
#   但对于像digit recognition那样的大型数据来说，我们应该尝试30~50层深度（或许还不足够
#   更应该画出学习曲线，来观察深度对模型的影响
rfc = RandomForestRegressor(n_estimators = 109,
                            random_state = 90)
GS = GridSearchCV(rfc, param_grid, cv = 3)
GS.fit(x_train, y_train)
print(GS.best_params_)
print(GS.best_score_)

param_grid = {"max_features": np.arange(5, 15, 1)}
rfc = RandomForestRegressor(n_estimators = 109,
                            random_state = 90)
GS = GridSearchCV(rfc, param_grid, cv = 3)
GS.fit(x_train, y_train)
print(GS.best_params_)
print(GS.best_score_)

param_grid = {'min_samples_leaf': np.arange(1, 1 + 10, 1)}
rfc = RandomForestRegressor(n_estimators = 109,
                            random_state = 90)
GS = GridSearchCV(rfc, param_grid, cv = 3)
GS.fit(x_train, y_train)
print(GS.best_params_)
print(GS.best_score_)

param_grid = {'min_samples_split': np.arange(2, 2 + 20, 1)}
rfc = RandomForestRegressor(n_estimators = 109,
                            random_state = 90)
GS = GridSearchCV(rfc, param_grid, cv = 3)
GS.fit(x_train, y_train)
print(GS.best_params_)
print(GS.best_score_)

##
# 最佳参数决策树
rfc = RandomForestRegressor(n_estimators = 109,
                            random_state = 90,
                            max_depth = 23,
                            max_features = 7,
                            min_samples_leaf = 5,
                            min_samples_split = 11,
                            criterion = 'mae')
score_pre = cross_val_score(rfc, x_train, y_train, cv = 10).mean()
print(score_pre)

##
# 拟合
rfc = rfc.fit(x_train, y_train)

##
# 导出模型
joblib.dump(rfc, "/Users/Bureaux/Documents/workspace/PyCharmProjects/BreastCancerMedicine/weights/t2.m")

##
# 训练
train_set_x, test_set_x = train_test_split(x_train, test_size = 0.2, shuffle = False)
train_set_y, test_set_y = train_test_split(y_train, test_size = 0.2, shuffle = False)
_, origin_y_test = train_test_split(era_activity.values, test_size = 0.2, shuffle = False)
print(train_set_x.shape)
print(test_set_x.shape)
print(train_set_y.shape)
print(test_set_y.shape)

rfc_split = RandomForestRegressor(n_estimators = 109,
                                  random_state = 90,
                                  max_depth = 21,
                                  max_features = 7,
                                  min_samples_leaf = 5,
                                  min_samples_split = 11,
                                  criterion = 'mae')

rfc_splite = rfc_split.fit(train_set_x, train_set_y)

##
# 测试集表现
split_y_train_out = rfc_splite.predict(test_set_x)
split_y_train_out = split_y_train_out.reshape(len(split_y_train_out), 1)
# split_y_train_out = y_scaler.inverse_transform(split_y_train_out)

print(split_y_train_out.shape)
print(origin_y_test.shape)

out_split = pd.DataFrame(split_y_train_out.reshape(len(split_y_train_out)), origin_y_test.reshape(len(origin_y_test)))
print(out_split.head(10))
print("Mean squared error: %.2f"
      % mean_squared_error(split_y_train_out.reshape(len(split_y_train_out)),
                           origin_y_test.reshape(len(origin_y_test))))

print('Variance score: %.2f' % r2_score(origin_y_test.reshape(len(origin_y_test)),
                                        split_y_train_out.reshape(len(split_y_train_out))
                                        ))

##
# 训练集表现
# clf_saved = joblib.load("/Users/Bureaux/Documents/workspace/PyCharmProjects/BreastCancerMedicine/weights/t2.m")
y_train_out = rfc_split.predict(x_train)
y_train_out = y_train_out.reshape(len(y_train_out), 1)
# y_train_out = y_scaler.inverse_transform(y_train_out)
print(y_train_out.shape)
print(era_activity.values.shape)
out = pd.DataFrame(era_activity.values.reshape(len(era_activity.values)), y_train_out.reshape(len(y_train_out)))
print(out.head(10))

print("Mean squared error: %.2f"
      % mean_squared_error(era_activity.values.reshape(len(era_activity.values)),
                           y_train_out.reshape(len(y_train_out))))

print('Variance score: %.2f' % r2_score(era_activity.values.reshape(len(era_activity.values)),
                                        y_train_out.reshape(len(y_train_out))))


##

joblib.dump(rfc_split, "/Users/Bureaux/Documents/workspace/PyCharmProjects/BreastCancerMedicine/weights/t2_rf.m")

##
# 导出
molecular_descriptor_test = pd.read_csv(path.molecular_descriptor_test_path)

molecular_descriptor_test = pd.DataFrame(molecular_descriptor_test,
                                         columns = ['SHsOH', 'SsOH', 'LipoaffinityIndex', 'C2SP2', 'BCUTc-1l',
                                                    'apol', 'minsOH', 'maxHsOH', 'MDEC-23', 'maxsssN', 'SwHBa',
                                                    'MLFER_A', 'ATSp4', 'SP-5', 'nHsOH', 'maxsOH', 'nC', 'nHBAcc',
                                                    'nAtom',
                                                    'minHsOH'])

print(molecular_descriptor_test.head())

print(molecular_descriptor_test.values.shape)

x_test = molecular_descriptor_test.values

y_pred = rfc.predict(x_test)
print(y_pred.shape)
print(y_pred)
y_pred = y_pred.reshape((len(y_pred), 1))

y_pred_df = pd.DataFrame(y_pred)
y_pred_df.to_csv(
    '/Users/Bureaux/Documents/workspace/PyCharmProjects/BreastCancerMedicine/data/export/output_xgboost.csv')
