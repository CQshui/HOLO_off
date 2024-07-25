# -*- coding: utf-8 -*-
"""
@ Time:     2024/7/14 14:54 2024
@ Author:   CQshui
$ File:     polar.py
$ Software: Pycharm
"""
import os
import numpy as np
import cv2
import time

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def read_image(folder_path):
    polar_0 = None
    polar_45 = None
    polar_90 = None
    polar_135 = None
    folders = os.listdir(folder_path)
    for folder in folders:
        # 判断是否为文件夹
        if os.path.isdir(os.path.join(folder_path, folder)):
            img_list = os.listdir(os.path.join(folder_path, folder))
            for img in img_list:
                if img != 'FFT.jpg' and img != 'phase_corrected.jpg':
                    if folder.startswith('Mono8_0_'):
                        polar_0 = cv2.imread(os.path.join(folder_path, folder, img), 0)
                    elif folder.startswith('Mono8_45_'):
                        polar_45 = cv2.imread(os.path.join(folder_path, folder, img), 0)
                    elif folder.startswith('Mono8_90_'):
                        polar_90 = cv2.imread(os.path.join(folder_path, folder, img), 0)
                    elif folder.startswith('Mono8_135_'):
                        polar_135 = cv2.imread(os.path.join(folder_path, folder, img), 0)

    return polar_0, polar_45, polar_90, polar_135

def Phase_Compensation_Zernike_Combination(aop_unwrapped, modes):
    print('> Phase Compensation ' + '............', end="")
    time_start = time.time()

    def Zernike_Polynomial(n, m, rho, phi):
        if m == 0:
            return np.sqrt(2 / (n + 1)) * rho ** n * np.cos(n * phi)
        elif m > 0:
            return np.sqrt(2 / (n + 1)) * rho ** n * np.cos(abs(m) * phi)
        else:
            # m < 0
            return np.sqrt(2 / (n + 1)) * rho ** n * np.sin(abs(m) * phi)

    '极坐标转换'
    height, width = aop.shape
    center_x, center_y = width // 2, height // 2
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # 极坐标：径向
    polar_r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    polar_r = polar_r / np.max(polar_r)  # 归一化半径
    # 极坐标：角向
    polar_phi = np.arctan2(y - center_y, x - center_x)

    '计算 Zernike 多项式'
    # modes = [(2, 0), (2, 2), (3, 1), (4, 0)]
    # modes = [(1, -1), (4, 0)]
    zernike_matrices = np.zeros((height, width, len(modes)))
    for i, (order, axis_order) in enumerate(modes):
        zernike_matrices[:, :, i] = Zernike_Polynomial(order, axis_order, polar_r, polar_phi)

    # 组合多项式
    zernike_matrix_combo = np.sum(zernike_matrices, axis=2)

    '计算Zernike系数'
    coefs, _, _, _ = np.linalg.lstsq(zernike_matrix_combo, aop_unwrapped, rcond=None)

    '计算补偿相位'
    phase_compensation_mat = np.dot(zernike_matrix_combo, coefs)

    '返回补偿后的相位'
    wave_phase_compensated = aop_unwrapped - phase_compensation_mat
    phase_processing = wave_phase_compensated

    print(' Finish. ', end="")
    time_end = time.time()
    print('Time Usage : {:.3f} /sec.'.format(time_end - time_start))

    show_phase = False
    if show_phase:

        plt.figure(figsize=(12.5, 4.5))
        plt.subplot(131)
        plt.imshow(aop_unwrapped)
        plt.title('Phase Origin')
        plt.colorbar()

        plt.subplot(132)
        plt.imshow(phase_compensation_mat)
        plt.title('Phase Compensation Matrix')
        plt.colorbar()

        plt.subplot(133)
        plt.imshow(phase_processing)
        plt.title('Phase Corrected')
        plt.colorbar()
        plt.show()

    return phase_processing


if __name__ == '__main__':
    # folder_path = r'F:\Data\20240717\baiyunmu\1\reconstruction_mine'
    folder_path = r'F:\Data\20240717\baiyunmu\1\reconstruction_mine'

    # 读取图像
    polar_imgs = read_image(folder_path)
    I_0 = np.array(polar_imgs[0], dtype=np.float32)
    I_45 = np.array(polar_imgs[1], dtype=np.float32)
    I_90 = np.array(polar_imgs[2], dtype=np.float32)
    I_135 = np.array(polar_imgs[3], dtype=np.float32)

    # 计算斯托克斯各分量
    S_0 = I_0 + I_90
    S_1 = I_0 - I_90
    S_2 = I_135 - I_45
    S_3 = np.zeros_like(S_0)

    show_S = False
    if show_S:
        plt.figure(figsize=(12.5, 4.5))
        plt.subplot(131)
        plt.imshow(S_0)
        plt.title('S0')
        plt.colorbar()

        plt.subplot(132)
        plt.imshow(S_1)
        plt.title('S1')
        plt.colorbar()

        plt.subplot(133)
        plt.imshow(S_2)
        plt.title('S2')
        plt.colorbar()
        plt.show()

    # 斯托克斯矢量
    S = np.stack((S_0, S_1, S_2, S_3))

    # DOPL
    dopl = np.divide(np.sqrt(S_1**2 + S_2**2), S_0)
    dopl = np.nan_to_num(dopl, posinf=0, neginf=0, nan=0)  # 将无效值替换为0\
    dopl = (dopl * 255).astype(np.uint8)  # 归一化并转换为8位整数
    cv2.imwrite(os.path.join(folder_path, 'DOPL.jpg'), dopl)
    # b, g, r = cv2.split(dopl)
    # img_plt = cv2.merge([r, g, b])
    plt.imsave(os.path.join(folder_path, 'DOPL_color.jpg'), dopl, cmap='PRGn')


    # AOP
    aop = np.arctan2(S_2, S_1) / 2
    aop1 = np.nan_to_num(aop, posinf=0, neginf=0, nan=0) # 将无效值替换为0
    # aop2 = Phase_Compensation_Zernike_Combination(aop1, modes = [(1, -1), (1, 1), (4, 0)])
    aop3 = ((aop1  + np.pi / 2 ) / np.pi * 255).astype(np.uint8)  # 将范围归一化到0到255，并转换为8位整数
    cv2.imwrite(os.path.join(folder_path, 'AOP.jpg'), aop3)
    plt.imsave(os.path.join(folder_path, 'AOP_color.jpg'), aop3, cmap='BrBG')

