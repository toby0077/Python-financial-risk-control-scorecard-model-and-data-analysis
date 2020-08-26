# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 22:42:47 2018

@author: Administrator
出现module 'xgboost' has no attribute 'DMatrix'的临时解决方法
初学者或者说不太了解Python才会犯这种错误，其实只需要注意一点！不要使用任何模块名作为文件名，任何类型的文件都不可以！我的错误根源是在文件夹中使用xgboost.*的文件名，当import xgboost时会首先在当前文件中查找，才会出现这样的问题。
        所以，再次强调：不要用任何的模块名作为文件名！
"""
import xgboost as xgb
from sklearn.cross_validation import train_test_split
import pandas as pd
import matplotlib.pylab as plt


#读取文件
readFileName="German_credit.xlsx"

#读取excel
df=pd.read_excel(readFileName)
list_columns=list(df.columns[:-1])
x=df.ix[:,:-1]
y=df.ix[:,-1]
names=x.columns

train_x, test_x, train_y, test_y=train_test_split(x,y,random_state=0)

dtrain=xgb.DMatrix(train_x,label=train_y)
dtest=xgb.DMatrix(test_x)

params={'booster':'gbtree',
    #'objective': 'reg:linear',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth':4,
    'lambda':10,
    'subsample':0.75,
    'colsample_bytree':0.75,
    'min_child_weight':2,
    'eta': 0.025,
    'seed':0,
    'nthread':8,
     'silent':1}

watchlist = [(dtrain,'train')]

bst=xgb.train(params,dtrain,num_boost_round=100,evals=watchlist)

ypred=bst.predict(dtest)

# 设置阈值, 输出一些评价指标
y_pred = (ypred >= 0.5)*1

#模型校验
from sklearn import metrics
print ('AUC: %.4f' % metrics.roc_auc_score(test_y,ypred))
print ('ACC: %.4f' % metrics.accuracy_score(test_y,y_pred))
print ('Recall: %.4f' % metrics.recall_score(test_y,y_pred))
print ('F1-score: %.4f' %metrics.f1_score(test_y,y_pred))
print ('Precesion: %.4f' %metrics.precision_score(test_y,y_pred))
metrics.confusion_matrix(test_y,y_pred)



print("xgboost:")  
#print("accuracy on the training subset:{:.3f}".format(bst.get_score(train_x,train_y)))
#print("accuracy on the test subset:{:.3f}".format(bst.get_score(test_x,test_y)))
print('Feature importances:{}'.format(bst.get_fscore()))


'''
AUC: 0.8135
ACC: 0.7640
Recall: 0.9641
F1-score: 0.8451
Precesion: 0.7523

#特征重要性和随机森林差不多
Feature importances:{'Account Balance': 80, 'Duration of Credit (month)': 119,
 'Most valuable available asset': 54, 'Payment Status of Previous Credit': 84, 
 'Value Savings/Stocks': 66, 'Age (years)': 94, 'Credit Amount': 149, 
 'Type of apartment': 20, 'Instalment per cent': 37,
 'Length of current employment': 70, 'Sex & Marital Status': 29,
 'Purpose': 67, 'Occupation': 13, 'Duration in Current address': 25,
 'Telephone': 15, 'Concurrent Credits': 23, 'No of Credits at this Bank': 7, 
 'Guarantors': 28, 'No of dependents': 6}
'''
