import os
from scipy.stats import pearsonr
import numpy as np
import openpyxl
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import cv2
from skimage import measure


def box_counting_dimension(image):
    """
    """
    # 盒子大小的集合
    box_sizes = np.array([2, 3, 4, 6, 8, 12, 16, 32])

    # 用于保存每个盒子大小的结果的数组
    box_counts = np.zeros(len(box_sizes))

    # 对于每个盒子大小
    for i in range(len(box_sizes)):
        box_size = box_sizes[i]

        # 计算在当前盒子大小下被覆盖的部分所占的比例
        num_box = 0
        for row in range(0, image.shape[0]-box_size, box_size):
            for col in range(0, image.shape[1]-box_size, box_size):
                if np.sum(image[row+box_size, col+box_size]) != 10000:
                    num_box += 1

        box_counts[i] = num_box
    # 拟合结果，得到盒维数
    box_dimension = np.log(box_counts) / np.log(box_sizes)

    # 计算斜率
    slope, intercept = np.polyfit(np.log(box_sizes), np.log(box_counts), 1)
    correlation, p_value = pearsonr(np.log(box_sizes), np.log(box_counts))

    return -slope


def output_xyl():
    # 创建一个工作簿
    wb = openpyxl.Workbook()
    ws = wb.active
    result = [num, Fr]
    for temp in result[:len(result)]:
        for num_value in range(len(temp)):
            ws.cell(row=num_value+1, column=result.index(temp)+1, value=temp[num_value])

    wb.save('C:\\Users\\d1009\\Desktop\\test\\fractal\\outputG.xlsx')

    return None


# img_path = 'C:\\Users\\d1009\\Desktop\\test\\generation\\Limestone\\'
img_path = 'C:\\Users\\d1009\\Desktop\\test\\generation\\Gypsum\\'
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
    _, binary_image = cv2.threshold(Z, 127, 255, cv2.THRESH_BINARY)
    # Z_not = cv2.bitwise_not(binary_image)
    Fr.append(box_counting_dimension(binary_image/255))
    num.append(number)

output_xyl()
