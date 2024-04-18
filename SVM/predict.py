# -*- coding: utf-8 -*-
"""
@ Time:     2024/4/18 23:14 2024
@ Author:   CQshui
$ File:     predict.py
$ Software: Pycharm
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pickle


# 加载模型
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# 加载数据集
df = pd.read_csv("svm_data.csv")
df = df.drop("id",axis=1)
sns.countplot(df["label"])
test = df

# 构建特征数组
features = ['area',
            'width'
]
test_X = test[features]

# 构建标签
test_y = test["label"]

# 数据归一化
ss = StandardScaler()
test_X = ss.fit_transform(test_X)

# 预测
predictions = model.predict(test_X)
print(predictions)