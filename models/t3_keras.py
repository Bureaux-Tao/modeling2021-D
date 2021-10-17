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

molecular_descriptor = pd.read_csv(path.molecular_descriptor_train_path)
molecular_descriptor = molecular_descriptor.drop(labels = 'SMILES', axis = 1)

admet = pd.read_csv(path.admet_train_path)
admet = admet.drop(labels = 'SMILES', axis = 1)

caco = pd.DataFrame(admet, columns = ['MN'])

print(molecular_descriptor.values.shape)

##
x_train_caco = molecular_descriptor.values
y_train_caco = caco.values

plt.rcParams['font.sans-serif'] = ['SF Mono']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 360  # 图片像素
plt.rcParams['figure.dpi'] = 360  # 分辨率

results = []

for i in range(20, 50, 1):
    n_components = i
    pca_model = PCA(n_components = n_components, whiten = True)
    x_pca_caco = pca_model.fit(x_train_caco).transform(x_train_caco)
    # print(x_pca_caco.shape)
    # print("降维后各主成分的方差值：", pca_model.explained_variance_)
    # print("降维后各主成分的方差值与总方差之比：", pca_model.explained_variance_ratio_)
    # print("奇异值分解后得到的特征值：", pca_model.singular_values_)
    # print("降维后主成分数：", pca_model.n_components_)
    train_caco_x, test_caco_x = train_test_split(x_pca_caco, test_size = 0.2, shuffle = False)
    train_caco_y, test_caco_y = train_test_split(y_train_caco, test_size = 0.2, shuffle = False)
    
    # print(train_caco_x.shape)
    # print(test_caco_x.shape)
    # print(train_caco_y.shape)
    # print(test_caco_y.shape)
    
    model = Sequential()
    model.add(Dense(8, input_dim = n_components, activation = 'relu', kernel_regularizer = regularizers.l2(0.01)))
    # model.add(Dropout(0.2))
    # model.add(Dense(16, activation = 'tanh', kernel_regularizer = regularizers.l2(0.01)))
    # model.add(Dropout(0.2))
    # model.add(Dense(16, activation = 'tanh', kernel_regularizer = regularizers.l2(0.01)))
    # model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'sigmoid', kernel_regularizer = regularizers.l2(0.01)))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    # model.summary()
    
    reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.5, patience = 3, verbose = 0, min_lr = 1e-6)
    early_stopping = EarlyStopping(monitor = 'loss', patience = 10, verbose = 0)  # 提前结束
    
    history = model.fit(train_caco_x, train_caco_y, validation_split = 0.2, epochs = 200, verbose = 0, batch_size = 16,
                        callbacks = [early_stopping, reduce_lr])
    # plt.plot(history.history['loss'], label = 'train')
    # plt.plot(history.history['val_loss'], label = 'test')
    # plt.legend()
    # plt.show()
    
    result_caco = model.evaluate(test_caco_x, test_caco_y)
    print('n_components:' + str(i))
    print(result_caco)
    results.append({'n_components': i, 'accuracy': result_caco})
print(results)
