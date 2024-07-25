# -*- coding: utf-8 -*-
"""
@ Time:     2024/7/18 14:05 2024
@ Author:   CQshui
$ File:     phase_compensation.py
$ Software: Pycharm
"""
import time
import numpy as np
import os
import cv2
from skimage.restoration import unwrap_phase

class Phase_compensation:
    def __init__(self, img_folder, save_folder):
        self.img_folder = img_folder
        self.save_folder = save_folder
        self.imgs = os.listdir(self.img_folder)


    def Phase_Compensation_Zernike_Combination(self, modes):
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
        coefs, _, _, _ = np.linalg.lstsq(zernike_matrix_combo, self.phase_processing, rcond=None)

        '计算补偿相位'
        self.phase_compensation_mat = np.dot(zernike_matrix_combo, coefs)
        # self.phase_compensation_mat = self.phase_compensation_mat / np.max(self.phase_compensation_mat)

        '返回补偿后的相位'
        self.wave_phase_compensated = self.phase_processing - self.phase_compensation_mat
        self.phase_processing = self.wave_phase_compensated

        print(' Finish. ', end="")
        time_end = time.time()
        print('Time Usage : {:.3f} /sec.'.format(time_end - time_start))


if __name__ == '__main__':
    Phase = Phase_compensation(r'', r'')
    for img in Phase.imgs:
        img_pth = os.path.join(Phase.img_folder, img)
        aop = cv2.imread(img_pth, 0)
        aop_unwrapped = unwrap_phase(aop)

