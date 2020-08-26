# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 21:57:29 2018

@author: Administrator
SVM需要标准化数据处理
"""
#标准化数据
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
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
X_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,random_state=42)
svm=SVC()
svm.fit(X_train,y_train)
print("accuracy on the training subset:{:.3f}".format(svm.score(X_train,y_train)))
print("accuracy on the test subset:{:.3f}".format(svm.score(x_test,y_test)))

'''
accuracy on the training subset:1.000
accuracy on the test subset:0.700

'''

#观察数据是否标准化
plt.plot(X_train.min(axis=0),'o',label='Min')
plt.plot(X_train.max(axis=0),'v',label='Max')
plt.xlabel('Feature Index')
plt.ylabel('Feature magnitude in log scale')
plt.yscale('log')
plt.legend(loc='upper right')

#标准化数据
X_train_scaled = preprocessing.scale(X_train)
x_test_scaled = preprocessing.scale(x_test)
svm1=SVC()
svm1.fit(X_train_scaled,y_train)
print("accuracy on the scaled training subset:{:.3f}".format(svm1.score(X_train_scaled,y_train)))
print("accuracy on the scaled test subset:{:.3f}".format(svm1.score(x_test_scaled,y_test)))
'''
accuracy on the scaled training subset:0.867
accuracy on the scaled test subset:0.800
'''



#改变C参数，调优,kernel表示核函数，用于平面转换，probability表示是否需要计算概率
svm2=SVC(C=10,gamma="auto",kernel='rbf',probability=True)
svm2.fit(X_train_scaled,y_train)
print("after c parameter=10,accuracy on the scaled training subset:{:.3f}".format(svm2.score(X_train_scaled,y_train)))
print("after c parameter=10,accuracy on the scaled test subset:{:.3f}".format(svm2.score(x_test_scaled,y_test)))
'''
after c parameter=10,accuracy on the scaled training subset:0.972
after c parameter=10,accuracy on the scaled test subset:0.716
'''


#计算样本点到分割超平面的函数距离
#print (svm2.decision_function(X_train_scaled))

#print (svm2.decision_function(X_train_scaled)[:20]>0)
#支持向量机分类
#print(svm2.classes_)

#malignant和bening概率计算,输出结果包括恶性概率和良性概率
#print(svm2.predict_proba(x_test_scaled))
#判断数据属于哪一类，0或1表示
#print(svm2.predict(x_test_scaled))





































