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
import numpy as np


# 加载模型
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# 加载数据集
df = pd.read_csv("F:/Data/Data/3vs1/output.csv")
df = df.drop("number",axis=1)
df.info()
df.head()
sns.countplot(df["label"])
test = df

# 构建特征数组
features = [# 'area',
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
test_X = test[features]

# 数据归一化
ss = StandardScaler()
test_X = ss.fit_transform(test_X)

# 预测
predictions = model.predict(test_X)
print(predictions)

# 计算纯度
width = test['width']
area = test['area']
types = predictions
volumn = np.array([[0], [0]], dtype=np.float64)
counts = np.array([[0], [0]], dtype=np.float64)
for item in range(len(types)):
    if types[item] == 1:    # gypsum
        counts[0] += 1
        volumn[0] += width[item]*area[item]
    else:                   # limestone
        counts[1] += 1
        volumn[1] += width[item]*area[item]

# 密度
density_G = 2.31
density_L = 2.71

# 石膏质量分数
purity = volumn[0]*density_G/(volumn[0]*density_G+volumn[1]*density_L)
print('purity =', purity[0])
print('count =', counts)

