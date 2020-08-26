# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 11:49:50 2018

@author: Administrator
神经网络需要预处理数据
"""
#Multi-layer Perceptron 多层感知机
from sklearn.neural_network import MLPClassifier
#标准化数据，否则神经网络结果不准确，和SVM类似
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import mglearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#读取文件
readFileName="German_credit.xlsx"

#读取excel
df=pd.read_excel(readFileName)
list_columns=list(df.columns[:-1])
x=df.ix[:,:-1]
y=df.ix[:,-1]
names=x.columns

#random_state 相当于随机数种子
x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,random_state=42)
mlp=MLPClassifier(random_state=42)
mlp.fit(x_train,y_train)
print("neural network:")    
print("accuracy on the training subset:{:.3f}".format(mlp.score(x_train,y_train)))
print("accuracy on the test subset:{:.3f}".format(mlp.score(x_test,y_test)))

scaler=StandardScaler()
x_train_scaled=scaler.fit(x_train).transform(x_train)
x_test_scaled=scaler.fit(x_test).transform(x_test)

mlp_scaled=MLPClassifier(max_iter=1000,random_state=42)
mlp_scaled.fit(x_train_scaled,y_train)
print("neural network after scaled:")    
print("accuracy on the training subset:{:.3f}".format(mlp_scaled.score(x_train_scaled,y_train)))
print("accuracy on the test subset:{:.3f}".format(mlp_scaled.score(x_test_scaled,y_test)))


mlp_scaled2=MLPClassifier(max_iter=1000,alpha=1,random_state=42)
mlp_scaled2.fit(x_train_scaled,y_train)
print("neural network after scaled and alpha change to 1:")    
print("accuracy on the training subset:{:.3f}".format(mlp_scaled2.score(x_train_scaled,y_train)))
print("accuracy on the test subset:{:.3f}".format(mlp_scaled2.score(x_test_scaled,y_test)))


#绘制颜色图,热图
plt.figure(figsize=(20,5))
plt.imshow(mlp_scaled.coefs_[0],interpolation="None",cmap="GnBu")
plt.yticks(range(30),names)
plt.xlabel("columns in weight matrix")
plt.ylabel("input feature")
plt.colorbar()

'''
neural network:
accuracy on the training subset:0.700
accuracy on the test subset:0.700
neural network after scaled:
accuracy on the training subset:1.000
accuracy on the test subset:0.704
neural network after scaled and alpha change to 1:
accuracy on the training subset:0.916
accuracy on the test subset:0.720
'''


