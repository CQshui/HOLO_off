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
                if img != 'FFT.jpg':
                    if folder.startswith('Mono8_0_'):
                        polar_0 = cv2.imread(os.path.join(folder_path, folder, img), 0)
                    elif folder.startswith('Mono8_45_'):
                        polar_45 = cv2.imread(os.path.join(folder_path, folder, img), 0)
                    elif folder.startswith('Mono8_90_'):
                        polar_90 = cv2.imread(os.path.join(folder_path, folder, img), 0)
                    elif folder.startswith('Mono8_135_'):
                        polar_135 = cv2.imread(os.path.join(folder_path, folder, img), 0)

    return polar_0, polar_45, polar_90, polar_135


if __name__ == '__main__':
    folder_path = r'F:\Data\20240713\baiyunmu\4\angular'
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

    # 斯托克斯矢量
    S = np.stack((S_0, S_1, S_2, S_3))

    # DOPL
    dopl = np.divide(np.sqrt(S_1**2 + S_2**2), S_0)
    dopl = np.nan_to_num(dopl, posinf=0, neginf=0, nan=0)  # 将无效值替换为0\
    dopl = (dopl * 255).astype(np.uint8)  # 归一化并转换为8位整数
    cv2.imwrite(os.path.join(folder_path, 'DOPL.jpg'), dopl)


    # AOP
    aop = np.arctan2(S_2, S_1) / 2
    aop1 = np.nan_to_num(aop, posinf=0, neginf=0, nan=0)  # 将无效值替换为0
    aop2 = ((aop1 + np.pi / 2) / np.pi * 255).astype(np.uint8)  # 将范围归一化到0到255，并转换为8位整数
    cv2.imwrite(os.path.join(folder_path, 'AOP.jpg'), aop2)

