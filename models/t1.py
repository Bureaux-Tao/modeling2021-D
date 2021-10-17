##
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import joblib
from utils import path

##

molecular_descriptor = pd.read_csv(path.molecular_descriptor_train_path)
molecular_descriptor = molecular_descriptor.drop(labels = 'SMILES', axis = 1)
era_activity = pd.read_csv(path.era_activity_train_path)
era_activity = era_activity.drop(labels = 'SMILES', axis = 1)

# print(molecular_descriptor.values.shape)
# print(era_activity.values.shape)

dataset_df = molecular_descriptor
dataset_df['pIC50'] = era_activity['pIC50']

min_max_scaler = preprocessing.MinMaxScaler()
dataset = min_max_scaler.fit_transform(dataset_df.values)
print(dataset.shape)

x_train = dataset[:, 0:dataset.shape[1] - 1]
y_train = dataset[:, -1]

print(x_train.shape)
print(y_train.shape)

##
plt.rcParams['font.sans-serif'] = ['SF Mono']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 360  # 图片像素
plt.rcParams['figure.dpi'] = 360  # 分辨率

param_grid = {'random_state': np.arange(80, 100, 1)}
rfc = RandomForestRegressor(n_estimators = 121,
                            random_state = 90)
GS = GridSearchCV(rfc, param_grid, cv = 2, verbose = 1)
GS.fit(x_train, y_train)

print(GS.best_params_)
print(GS.best_score_)

##

param_grid = {'n_estimators': np.arange(110, 130, 1)}
rfc = RandomForestRegressor(n_estimators = 121,
                            random_state = 90)
GS = GridSearchCV(rfc, param_grid, cv = 2, verbose = 1, scoring = 'neg_mean_squared_error')
GS.fit(x_train, y_train)

print(GS.best_params_)
print(GS.best_score_)

param_grid = {'max_depth': np.arange(10, 30, 1)}
rfc = RandomForestRegressor(n_estimators = 121,
                            random_state = 90)
GS = GridSearchCV(rfc, param_grid, cv = 2, verbose = 1, scoring = 'neg_mean_squared_error')
GS.fit(x_train, y_train)

print(GS.best_params_)
print(GS.best_score_)

param_grid = {'max_leaf_nodes': np.arange(20, 40, 1)}
#   一般根据数据的大小来进行一个试探，乳腺癌数据很小，所以可以采用1~10，或者1~20这样的试探
#   但对于像digit recognition那样的大型数据来说，我们应该尝试30~50层深度（或许还不足够
#   更应该画出学习曲线，来观察深度对模型的影响
rfc = RandomForestRegressor(n_estimators = 121,
                            random_state = 90)
GS = GridSearchCV(rfc, param_grid, cv = 2, scoring = 'neg_mean_squared_error')
GS.fit(x_train, y_train)
print(GS.best_params_)
print(GS.best_score_)

param_grid = {'min_samples_split': np.arange(2, 2 + 20, 1)}
rfc = RandomForestRegressor(n_estimators = 121,
                            random_state = 90)
GS = GridSearchCV(rfc, param_grid, cv = 2, scoring = 'neg_mean_squared_error')
GS.fit(x_train, y_train)

print(GS.best_params_)
print(GS.best_score_)

param_grid = {'min_samples_leaf': np.arange(1, 1 + 10, 1)}
rfc = RandomForestRegressor(n_estimators = 121,
                            random_state = 90)
GS = GridSearchCV(rfc, param_grid, cv = 2, scoring = 'neg_mean_squared_error')
GS.fit(x_train, y_train)

print(GS.best_params_)
print(GS.best_score_)

param_grid = {'max_features': np.arange(15, 35, 1)}
rfc = RandomForestRegressor(n_estimators = 121,
                            random_state = 90)
GS = GridSearchCV(rfc, param_grid, cv = 2, scoring = 'neg_mean_squared_error')
GS.fit(x_train, y_train)

print(GS.best_params_)
print(GS.best_score_)

##
rfc = RandomForestRegressor(n_estimators = 111,
                            random_state = 92,
                            max_depth = 26,
                            max_leaf_nodes = 39,
                            max_features = 31,
                            min_samples_leaf = 3,
                            min_samples_split = 2)
score_pre = cross_val_score(rfc, x_train, y_train, cv = 5).mean()
print(score_pre)
##
rfc = rfc.fit(x_train, y_train)

##

joblib.dump(rfc, "/Users/Bureaux/Documents/workspace/PyCharmProjects/BreastCancerMedicine/weights/t1_new.m")
##
permutation_weight = rfc.feature_importances_  # 每个特征的重要程度
print(len(permutation_weight))

head = molecular_descriptor.columns
head = head[0: len(head) - 1]
print(len(head))

permutation_importance = {}
for i in range(0, len(permutation_weight)):
    permutation_importance[head[i]] = permutation_weight[i]

# print(permutation_importance)

permutation_importance_result = sorted(permutation_importance.items(), key = lambda x: x[1], reverse = True)

# with open('../data/export/permutation_importance_result.txt', 'a') as f:
for i in range(20):
    print(permutation_importance_result[i])
    # f.writelines(str(permutation_importance_result[i][0]))
    # f.writelines('\t')
    # f.writelines(str(permutation_importance_result[i][1]))
    # f.writelines('\n')

# print(rfc.apply(x_train))  # 每个样本在每个树中的节点的索引
# print(rfc.predict(Xtest))  # 样本点在每个结果的可能性
# print(rfc.predict_proba(Xtest))  # 样本的平均概率

##
#
# clf_saved = joblib.load("../weights/t1.m")
#
# molecular_descriptor_test = pd.read_csv(path.molecular_descriptor_test_path)
# molecular_descriptor_test = molecular_descriptor_test.drop(labels = 'SMILES', axis = 1)
#
# x_test = min_max_scaler.fit_transform(molecular_descriptor_test)
#
# y_pred = clf_saved.predict(x_test)
# y_pred = y_pred.reshape((len(y_pred), 1))
#
#
# # data_pred = np.append(x_test, y_pred, axis = 1)
# y_pred = min_max_scaler.inverse_transform(y_pred)
#
# y_pred_df = pd.DataFrame(y_pred)
# y_pred_df.to_csv('../data/export/output.csv')
