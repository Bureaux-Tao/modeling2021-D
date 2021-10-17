##

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import joblib
from copy import copy

from utils import path

##

all_data_df = pd.read_csv(path.all_data_path)
all_data_df = all_data_df.drop(labels = 'SMILES', axis = 1)

pIC50_df = all_data_df['pIC50']
pIC50 = pIC50_df.values

admet_df = all_data_df['ADMET']
admet = admet_df.values

all_data_df = all_data_df.drop(labels = 'pIC50', axis = 1)
all_data_df = all_data_df.drop(labels = 'ADMET', axis = 1)

all_data = all_data_df.values

x_train, x_test, y_train, y_test = train_test_split(all_data, admet, test_size = 0.2, shuffle = False)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

plt.rcParams['font.sans-serif'] = ['SF Mono']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 360  # 图片像素
plt.rcParams['figure.dpi'] = 360  # 分辨率

##
param_grid = {'n_estimators': np.arange(40, 60, 1)}
rfc = RandomForestClassifier(n_estimators = 121,
                             random_state = 90)
GS = GridSearchCV(rfc, param_grid, cv = 5, verbose = 1)
GS.fit(x_train, y_train)

print(GS.best_params_)
print(GS.best_score_)

##
param_grid = {'random_state': np.arange(150, 170, 1)}
rfc = RandomForestClassifier(n_estimators = 121,
                             random_state = 90)
GS = GridSearchCV(rfc, param_grid, cv = 5, verbose = 1)
GS.fit(x_train, y_train)

print(GS.best_params_)
print(GS.best_score_)

##
param_grid = {'max_depth': np.arange(1, 20, 1)}
rfc = RandomForestClassifier(n_estimators = 121,
                             random_state = 90)
GS = GridSearchCV(rfc, param_grid, cv = 5, verbose = 1)
GS.fit(x_train, y_train)

print(GS.best_params_)
print(GS.best_score_)

param_grid = {'max_leaf_nodes': np.arange(20, 40, 1)}
#   一般根据数据的大小来进行一个试探，乳腺癌数据很小，所以可以采用1~10，或者1~20这样的试探
#   但对于像digit recognition那样的大型数据来说，我们应该尝试30~50层深度（或许还不足够
#   更应该画出学习曲线，来观察深度对模型的影响
rfc = RandomForestClassifier(n_estimators = 121,
                             random_state = 90)
GS = GridSearchCV(rfc, param_grid, cv = 5)
GS.fit(x_train, y_train)
print(GS.best_params_)
print(GS.best_score_)

param_grid = {'criterion': ['gini', 'entropy']}
rfc = RandomForestClassifier(n_estimators = 121,
                             random_state = 90)
GS = GridSearchCV(rfc, param_grid, cv = 5)
GS.fit(x_train, y_train)

print(GS.best_params_)
print(GS.best_score_)

param_grid = {'min_samples_split': np.arange(2, 2 + 20, 1)}
rfc = RandomForestClassifier(n_estimators = 121,
                             random_state = 90)
GS = GridSearchCV(rfc, param_grid, cv = 5)
GS.fit(x_train, y_train)

print(GS.best_params_)
print(GS.best_score_)

param_grid = {'min_samples_leaf': np.arange(1, 1 + 10, 1)}
rfc = RandomForestClassifier(n_estimators = 121,
                             random_state = 90)
GS = GridSearchCV(rfc, param_grid, cv = 5)
GS.fit(x_train, y_train)

print(GS.best_params_)
print(GS.best_score_)

param_grid = {'max_features': np.arange(5, 25, 1)}
rfc = RandomForestClassifier(n_estimators = 121,
                             random_state = 90)
GS = GridSearchCV(rfc, param_grid, cv = 5)
GS.fit(x_train, y_train)

print(GS.best_params_)
print(GS.best_score_)

##
rfc = RandomForestClassifier(n_estimators = 50, random_state = 161, max_depth = 2, max_leaf_nodes = 29,
                             criterion = 'entropy', min_samples_split = 7, max_features = 13)
score = cross_val_score(rfc, x_train, y_train, cv = 5).mean()
print(score)

##
rfc = rfc.fit(x_train, y_train)

##
y_pred = rfc.predict(x_test)
print(accuracy_score(y_pred, y_test))

##
joblib.dump(rfc, "/Users/Bureaux/Documents/workspace/PyCharmProjects/BreastCancerMedicine/weights/t4.m")

##
permutation_weight = rfc.feature_importances_  # 每个特征的重要程度
print(len(permutation_weight))

head = all_data_df.columns
# head = head[0: len(head) - 1]
print(len(head))
#
permutation_importance = {}
for i in range(0, len(permutation_weight)):
    permutation_importance[head[i]] = permutation_weight[i]

# print(permutation_importance)

permutation_importance_result = sorted(permutation_importance.items(), key = lambda x: x[1], reverse = True)

with open(
        '/Users/Bureaux/Documents/workspace/PyCharmProjects/BreastCancerMedicine/data/statistics/permutation_importance_result_t4.txt',
        'a') as f:
    for i in range(20):
        print(permutation_importance_result[i])
        f.writelines(str(permutation_importance_result[i][0]))
        f.writelines('\t')
        f.writelines(str(permutation_importance_result[i][1]))
        f.writelines('\n')

##
pd.set_option('display.max_columns', None)
new_df = pd.DataFrame(all_data_df,
                      columns = ['ATSc2', 'ETA_Eta_R', 'ETA_Eta_R_L', 'ECCEN', 'apol', 'ETA_Eta_F_L', 'MLFER_L',
                                 'ATSm3', 'SP-0', 'Kier2', 'SHother', 'maxssssC', 'SP-1', 'ETA_Shape_X', 'bpol',
                                 'MLFER_BH', 'C2SP2', 'SP-2', 'SPC-6', 'VC-4'])

new_df['pIC50'] = pIC50_df
# print(new_df.head())

filtered_df = new_df[new_df['pIC50'] > 9.0]
# filtered_df.to_csv(
#     '/Users/Bureaux/Documents/workspace/PyCharmProjects/BreastCancerMedicine/data/export/o.csv')
print(filtered_df.info)

##
iforest = IsolationForest(n_estimators = 100,
                          contamination = 0.1,
                          bootstrap = False, n_jobs = -1)

X = filtered_df.drop(labels = 'pIC50', axis = 1).values
print(X.shape)

##
pred = iforest.fit_predict(X)
scores = iforest.decision_function(X)
print(pred)
# print(scores)
if_out = pd.DataFrame(pred)
if_out.columns = ['pred']
# filtered_df = pd.concat([filtered_df, if_out], axis = 1)
# filtered_df['scores'] = pd.DataFrame(iforest.decision_function(X))
# filtered_df['anomaly_label'] = pd.DataFrame(pred)
filtered_df.to_csv(
    '/Users/Bureaux/Documents/workspace/PyCharmProjects/BreastCancerMedicine/data/export/filtered_t4.csv')
if_out.to_csv(
    '/Users/Bureaux/Documents/workspace/PyCharmProjects/BreastCancerMedicine/data/export/if_t4.csv')

##
output_df = pd.read_csv(
    '/Users/Bureaux/Documents/workspace/PyCharmProjects/BreastCancerMedicine/data/export/filtered_t4.csv')

output_df = output_df[output_df.pred == 1]
output_df = output_df.drop(labels = 'id', axis = 1)
output_df = output_df.drop(labels = 'pred', axis = 1)
print(output_df.head())
##
info = []
for index, column in output_df.iteritems():
    info.append({
        'column': index,
        'max': np.max(column.values),
        'min': np.min(column.values)
    })
print(info)
