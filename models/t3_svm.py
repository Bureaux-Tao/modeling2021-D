##
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from utils import path

##

molecular_descriptor = pd.read_csv(path.molecular_descriptor_train_path)
molecular_descriptor = molecular_descriptor.drop(labels = 'SMILES', axis = 1)

admet = pd.read_csv(path.admet_train_path)
admet = admet.drop(labels = 'SMILES', axis = 1)

# para = 'Caco-2'
# para = 'CYP3A4'
# para = 'hERG'
# para = 'HOB'
para = 'MN'
data = pd.DataFrame(admet, columns = [para])

print(molecular_descriptor.values.shape)

##
x_train = molecular_descriptor.values
y_train = data.values.reshape(len(data.values))
max_method = ''
max_acc = 0
acc_linear = []
acc_poly = []
acc_rbf = []
for i in range(20, 80):
    n_components = i
    pca_model = PCA(n_components = n_components, whiten = True)
    x_pca = pca_model.fit(x_train).transform(x_train)
    # print("降维后各主成分的方差值：", pca_model.explained_variance_)
    # print("降维后各主成分的方差值与总方差之比：", pca_model.explained_variance_ratio_)
    # print("奇异值分解后得到的特征值：", pca_model.singular_values_)
    # print("降维后主成分数：", pca_model.n_components_)
    train_x, test_x = train_test_split(x_pca, test_size = 0.2, shuffle = False)
    train_y, test_y = train_test_split(y_train, test_size = 0.2, shuffle = False)
    
    print('\nn_components:' + str(i))
    for kernel in ['linear', 'poly', 'rbf']:
        svm = SVC(kernel = kernel)
        svm.fit(train_x, train_y)
        y_pred = svm.predict(test_x)
        if accuracy_score(y_pred, test_y) > max_acc:
            max_acc = accuracy_score(y_pred, test_y)
            max_method = 'n_components:' + str(i) + "  kernel:" + kernel
        if kernel == 'linear':
            acc_linear.append(accuracy_score(y_pred, test_y))
        elif kernel == 'poly':
            acc_poly.append(accuracy_score(y_pred, test_y))
        else:
            acc_rbf.append(accuracy_score(y_pred, test_y))
        # print(kernel + ":" + str(accuracy_score(y_pred, test_y)))

print('linear:')
print(acc_linear)
print('poly:')
print(acc_poly)
print('rbf:')
print(acc_rbf)
print(max_method)
print(max_acc)

##
x_train = molecular_descriptor.values
y_train = data.values.reshape(len(data.values))
pca_model = PCA(n_components = 24, whiten = True)
x_pca = pca_model.fit(x_train).transform(x_train)
train_x, test_x = train_test_split(x_pca, test_size = 0.2, shuffle = False)
train_y, test_y = train_test_split(y_train, test_size = 0.2, shuffle = False)

max_acc = 0
acc_c = []
max_i = 0
for i in range(1, 20):
    svm = SVC(kernel = 'linear', C = i / 10)
    svm.fit(train_x, train_y)
    y_pred = svm.predict(test_x)
    acc_c.append(accuracy_score(y_pred, test_y))
    if accuracy_score(y_pred, test_y) > max_acc:
        max_acc = accuracy_score(y_pred, test_y)
        max_i = i / 10
    # print(accuracy_score(y_pred, test_y))
print(acc_c)
print(max_i)
print(max_acc)

##
x_train = molecular_descriptor.values
y_train = data.values.reshape(len(data.values))
pca_model = PCA(n_components = 24, whiten = True)
x_pca = pca_model.fit(x_train).transform(x_train)
train_x, test_x = train_test_split(x_pca, test_size = 0.2, shuffle = False)
train_y, test_y = train_test_split(y_train, test_size = 0.2, shuffle = False)
svm = SVC(kernel = 'linear', C = 0.4)
svm.fit(train_x, train_y)
y_pred = svm.predict(test_x)
print(accuracy_score(y_pred, test_y))
print(precision_score(y_pred, test_y))
print(recall_score(y_pred, test_y))
print(f1_score(y_pred, test_y))
# joblib.dump(svm, "/Users/Bureaux/Documents/workspace/PyCharmProjects/BreastCancerMedicine/weights/t3_" + para + ".m")

##
molecular_descriptor_test = pd.read_csv(path.molecular_descriptor_test_path)
molecular_descriptor_test = molecular_descriptor_test.drop(labels = 'SMILES', axis = 1)
pred_x = molecular_descriptor_test.values
x_pca_pred = pca_model.fit(pred_x).transform(pred_x)
print(svm.predict(x_pca_pred))
y_pred_df = pd.DataFrame(svm.predict(x_pca_pred))
y_pred_df.columns = [para]
y_pred_df.to_csv('/Users/Bureaux/Documents/workspace/PyCharmProjects/BreastCancerMedicine/data/export/' + para + '.csv')

##
# hERG/HOB - special
molecular_descriptor_test = pd.read_csv(path.molecular_descriptor_test_path)
molecular_descriptor_test = molecular_descriptor_test.drop(labels = 'SMILES', axis = 1)
pred_x = molecular_descriptor_test.values
print(pred_x.shape)
pred_x = np.vstack((pred_x, pred_x))
print(pred_x.shape)

x_pca_pred = pca_model.fit(pred_x).transform(pred_x)
out = svm.predict(x_pca_pred)
print(out.shape)
out = out[:int(out.shape[0]/2)]
print(out.shape)
y_pred_df = pd.DataFrame(out)
y_pred_df.columns = [para]
y_pred_df.to_csv('/Users/Bureaux/Documents/workspace/PyCharmProjects/BreastCancerMedicine/data/export/' + para + '.csv')
