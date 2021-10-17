##
import numpy as np
import pandas as pd
from keras import Sequential, regularizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Dropout
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
import matplotlib.pyplot as plt
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
print(x_train)
print(x_train.shape)

y_scaler = preprocessing.MinMaxScaler()
y_train = y_scaler.fit_transform(y_train)
# y_train = y_train.reshape(len(y_train))
print(y_train)
print(y_train.shape)

##
# x_train_caco = molecular_descriptor.values
# y_train_caco = caco.values
#
plt.rcParams['font.sans-serif'] = ['SF Mono']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 360  # 图片像素
plt.rcParams['figure.dpi'] = 360  # 分辨率
#
# results = []
#
# for i in range(20, 50, 1):
#     n_components = i
#     pca_model = PCA(n_components = n_components, whiten = True)
#     x_pca_caco = pca_model.fit(x_train_caco).transform(x_train_caco)
#     # print(x_pca_caco.shape)
#     # print("降维后各主成分的方差值：", pca_model.explained_variance_)
#     # print("降维后各主成分的方差值与总方差之比：", pca_model.explained_variance_ratio_)
#     # print("奇异值分解后得到的特征值：", pca_model.singular_values_)
#     # print("降维后主成分数：", pca_model.n_components_)
train_x, test_x = train_test_split(x_train, test_size = 0.2, shuffle = False)
train_y, test_y = train_test_split(y_train, test_size = 0.2, shuffle = False)
_, _, y_train_o, y_test_o = train_test_split(original_x, original_y, test_size = 0.2, shuffle = False)
#
print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)
#
model = Sequential()
model.add(Dense(16, input_dim = train_x.shape[1], activation = 'tanh', kernel_regularizer = regularizers.l2(0.01)))
# model.add(Dropout(0.2))
# model.add(Dense(16, activation = 'tanh', kernel_regularizer = regularizers.l2(0.01)))
# model.add(Dropout(0.2))
model.add(Dense(8, activation = 'tanh', kernel_regularizer = regularizers.l2(0.01)))
# model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
model.summary()
#
reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.5, patience = 3, verbose = 1, min_lr = 1e-8)
early_stopping = EarlyStopping(monitor = 'loss', patience = 10, verbose = 1)  # 提前结束
#
history = model.fit(train_x, train_y, validation_split = 0.2, epochs = 200, verbose = 2, batch_size = 16,
                    callbacks = [early_stopping, reduce_lr])
plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'test')
plt.legend()
plt.show()

##
pred = model.predict(train_x)
pred = y_scaler.inverse_transform(pred.reshape(len(pred), 1))
pred = pred.reshape(len(pred))
# print(len(pred))
# print(len(y_test_o))
print('training:')
print("Mean squared error: %.2f"
      % mean_squared_error(pred, y_train_o))
print('Variance score: %.2f' % r2_score(pred, y_train_o))

print('testing:')
pred = model.predict(test_x)
pred = y_scaler.inverse_transform(pred.reshape(len(pred), 1))
pred = pred.reshape(len(pred))
# print(len(pred))
# print(len(y_test_o))
print("Mean squared error: %.2f"
      % mean_squared_error(pred, y_test_o))
print('Variance score: %.2f' % r2_score(pred, y_test_o))
