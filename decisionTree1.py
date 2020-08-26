# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 22:14:10 2018

@author: Administrator
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
import pydotplus 
from IPython.display import Image
import graphviz
from sklearn.tree import export_graphviz
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

trees=1000
#读取文件
readFileName="German_credit.xlsx"

#读取excel
df=pd.read_excel(readFileName)
list_columns=list(df.columns[:-1])
x=df.ix[:,:-1]
y=df.ix[:,-1]
names=x.columns

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)

best_depth=4
print("best_depth:",best_depth)

best_tree= DecisionTreeClassifier(max_depth=best_depth,random_state=0)
best_tree.fit(x_train,y_train)
accuracy_training=best_tree.score(x_train,y_train)
accuracy_test=best_tree.score(x_test,y_test)
print("decision tree:")    
print("accuracy on the training subset:{:.3f}".format(best_tree.score(x_train,y_train)))
print("accuracy on the test subset:{:.3f}".format(best_tree.score(x_test,y_test)))

n_features=x.shape[1]
plt.barh(range(n_features),best_tree.feature_importances_,align='center')
plt.yticks(np.arange(n_features),names)
plt.title("Decision Tree:")
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()

#生成一个dot文件，以后用cmd形式生成图片
export_graphviz(best_tree,out_file="creditTree.dot",class_names=['bad','good'],feature_names=names,impurity=False,filled=True)



'''
best_depth: 4
decision tree:
accuracy on the training subset:0.760
accuracy on the test subset:0.692
'''



