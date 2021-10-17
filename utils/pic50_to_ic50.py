import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame
import math

df = pd.read_csv("output.csv")
list = df['data'].tolist()
def IC50(PIC50):
    IC50 = []
    for i in range(0,len(PIC50)):
        temp = math.pow(10,(9-PIC50[i]))
        IC50.append(temp)
    return IC50
x= IC50(list)
np.savetxt('new.csv', x, delimiter = ',')