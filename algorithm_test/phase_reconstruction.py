import numpy as np
from matplotlib import pyplot as plt
from skimage.restoration import unwrap_phase, denoise_bilateral, inpaint
# from test import unwrap_phase
from skimage.filters import gaussian
# 用高斯滤波来进行相位补偿，sigma用了定值，效果不是很好，对颗粒相位的影响很大，部分小颗粒甚至被滤掉了
# 把频谱严格移到中心后似乎不再需要相位补偿，对搭建的第二个离轴系统是如此。可以考虑把高斯滤波相位补偿作为可选的辅助功能并入程序。


def find_max_min(arr):
    max_val = float('-inf')  # 初始化为负无穷大
    min_val = float('+inf')  # 初始化为正无穷大
    for row in arr:
        for num in row:
            if num > max_val:
                max_val = num
            if num < min_val:
                min_val = num
    return max_val, min_val


def phase_unwrap(U):
    ang = np.angle(U)
    phase_unwrapped = unwrap_phase(ang)
    # result = find_max_min(phase_unwrapped)
    # print("最大值：", result[0])
    # print("最小值：", result[1])
    return phase_unwrapped


def compensate(U, height, width):
    denoised = gaussian(U, sigma=500)
    # ang = np.angle(U)
    # k = np.zeros((height, width), dtype=float)
    # for j in range(1, width):
    #     for h in range(2, height):
    #         if (ang[h, j] - ang[h - 1, j]) >= np.pi:
    #             k[h, j] = k[h - 1, j] - 1
    #         elif abs(ang[h, j] - ang[h - 1, j]) < np.pi:
    #             k[h, j] = k[h - 1, j]
    #         elif (ang[h, j] - ang[h - 1, j]) < (-np.pi):
    #             k[h, j] = k[h - 1, j] + 1
    # for h in range(1, height):
    #     for p in range(2, width):
    #         if (ang[h, p] - ang[h, p-1]) >= np.pi:
    #             k[h, p] = k[h, p-1] - 1
    #         elif abs(ang[h, p] - ang[h, p-1]) < np.pi:
    #             k[h, p] = k[h, p-1] - 1
    #         elif (ang[h, p] - ang[h, p-1]) < (-np.pi):
    #             k[h, p] = k[h, p-1] + 1
    return denoised
