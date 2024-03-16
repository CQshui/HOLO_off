import os

import numpy as np
import openpyxl
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import cv2


def box_count(Z, k):
    """计算所有盒子的数量"""
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
        np.arange(0, Z.shape[1], k), axis=1)

    # 把每个大于0的格子视作非空
    return len(np.where((S > 0))[0])


def fractal_dimension(Z):
    """使用盒计数法计算分形维数"""
    p = min(Z.shape)
    n = 2**np.floor(np.log(p)/np.log(2))
    n = int(np.log(n)/np.log(2))
    sizes = 2**np.arange(n, 1, -1)

    # 计算所有盒子的数量
    counts = []
    for size in sizes:
        counts.append(box_count(Z, size))

    # 计算斜率 （因为在对数图上的线性回归对应着幂律）
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]


def output_xyl():
    # 创建一个工作簿
    wb = openpyxl.Workbook()
    ws = wb.active
    result = [num, Fr]
    for temp in result[:len(result)]:
        for num_value in range(len(temp)):
            ws.cell(row=num_value+1, column=result.index(temp)+1, value=temp[num_value])

    wb.save('D:\\Desktop\\test\\fractal\\output.xlsx')

    return None


# img_path = 'D:\\Desktop\\test\\generation\\Limestone\\'
img_path = 'D:\\Desktop\\test\\generation\\Gypsum\\'
file_names = os.listdir(img_path)
image_files = [f for f in file_names if any(ext in f.lower() for ext in ('.jpg', '.jpeg', '.png', '.bmp'))]
Fr = ['FR']
num = ['NUM']
number = 0

for file_name in image_files:
    number += 1
    # 构建完整的文件路径
    image_path = os.path.join(img_path, file_name)
    Z = cv2.imread(image_path, 0)
    Fr.append(fractal_dimension(Z))
    num.append(number)

output_xyl()
