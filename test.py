# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 19:36:29 2018

@author: Administrator
#乳腺癌数据测试中，卡方检验效果不如随机森林,卡方筛选的2个最好因子在随机森林中权重不是最大

"""
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
cancer = load_breast_cancer()
X= cancer.data
y= cancer.target

#特征工程-卡方筛选K个最佳特征
#乳腺癌数据测试中，卡方检验效果不如随机森林,卡方筛选的2个最好因子在随机森林中权重不是最大
#we can perform a \chi^2 test to the samples to retrieve only the two best features as follows:
X_chi2 = SelectKBest(chi2, k=2).fit_transform(X, y)
print("X_chi2 shapes:",X_chi2.shape) 

##特征工程-支持向量l1
#根据模型选择最佳特征，线性svc选出5个最佳特征
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_lsvc1 = model.transform(X)
print("LinearSVC l1 shapes:",X_lsvc1.shape) 

#特征工程-支持向量l2
lsvc = LinearSVC(C=0.01, penalty="l2", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_lsvc2 = model.transform(X)
print("LinearSVC l2 shapes:",X_lsvc2.shape) 

#特征工程-forest of trees 随机森林
trees = ExtraTreesClassifier(n_estimators=10000)
trees = trees.fit(X, y)
trees.feature_importances_  
model = SelectFromModel(trees, prefit=True)
X_trees = model.transform(X)
print("forest trees shapes:",X_trees.shape) 

'''
测试结果随机森林特征工程效果最好，其次是支持向量l2模式
X_chi2 shapes: (569, 2)
LinearSVC l1 shapes: (569, 5)
LinearSVC l2 shapes: (569, 9)
forest trees shapes: (569, 10)
'''

#主成分分析
pca =PCA(n_components=0.98)
#pca= PCA(n_components='mle')

digits = load_breast_cancer()
X_digits = cancer.data
y_digits = digits.target

#scaled_x=preprocessing.scale(X_digits)
# Plot the PCA spectrum
#pca.fit(scaled_x)
pca.fit(X_digits)
print("PCA analysis:")
print (pca.explained_variance_ratio_)
print (pca.explained_variance_)
print (pca.n_components_)

#计算协方差
pca.get_covariance()
#Estimated precision of data.计算数据估计的准确性
pca.get_precision()
              