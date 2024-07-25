# -*- coding: utf-8 -*-
"""
@ Time:     2024/4/18 22:19 2024
@ Author:   CQshui
$ File:     svm.py
$ Software: Pycharm
"""
# import numpy as np
# from sklearn.svm import SVC
#
# x = np.array([[1, -1], [-2, -1], [1, 1], [2, 1]])
# y = np.array([1, 1, 2, 2])
# clf = SVC()
# clf.fit(x, y)
# print(clf.predict([[1, 1], [1, -3]]))

import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import pickle

df = pd.read_csv(r"I:\0_Datas\se_svm\all.csv")
df = df.drop("number",axis=1)
df.info()
df.head()
sns.countplot(df["label"])

"""
4  数据分割
"""

train,test = train_test_split(df,test_size=0.3)

# 构建训练集 测试集 特征数组
features = ['area',
            'aspect_ratio',
            'roundness',
            'solid',
            # 'gray1',
            # 'gray2',
            # 'gray3',
            # 'contrast',
            # 'ASM',
            # 'correlation',
            # 'entropy',
            'Hu1',
            'Hu2',
            'Hu3',
            'Hu4',
            'Hu5',
            'Hu6',
            'Hu7'
            ]
train_X = train[features]
test_X = test[features]

# 构建 标签
train_y = train["label"]
test_y = test["label"]

"""
5  数据归一化
"""
ss = StandardScaler()
train_X = ss.fit_transform(train_X)
test_X = ss.fit_transform(test_X)


"""
6  模型训练及预测
"""
model = SVC()
model.fit(train_X,train_y)

predictions = model.predict(test_X)
# print(predictions)


"""
7 模型评价
"""
accuracy_score(test_y,predictions)        # 0.7775100401606426  #sediment 0.3405067684831656print(accuracy_score(test_y,predictions))


# save model
with open(r'I:\0_Datas\se_svm\model.pkl', 'wb') as file:
    pickle.dump(model, file)
