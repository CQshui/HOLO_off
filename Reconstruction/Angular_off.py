# -*- coding: utf-8 -*-
"""
@ Time:     2024/4/27 10:30 2024
@ Author:   CQshui
$ File:     Angular_off.py
$ Software: Pycharm
"""
import glob
import os
import cv2
import numpy as np
from PIL import Image
from numpy.fft import fftshift, fft2, ifft2, ifftshift
import matplotlib.pyplot as plt
from phase_reconstruction import phase_unwrap


# 对FFT频谱图截图
def cut(img_path):
    global img, img_height, img_width
    img = cv2.imread(img_path, 0)
    img_height, img_width = img.shape[:2]
    cv2.namedWindow('image', 0)
    cv2.setMouseCallback('image', on_mouse)
    cv2.imshow('image', img)
    cv2.waitKey(0)


def on_mouse(event, x, y, flags, param):
    global point1, point2, U0
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        point1 = (x, y)
        cv2.circle(img2, point1, 10, (0, 255, 0), 3)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
        cv2.rectangle(img2, point1, (x, y), (255, 0, 0), 3)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
        point2 = (x, y)
        cv2.rectangle(img2, point1, point2, (0, 0, 255), 8)
        cv2.imshow('image', img2)
        min_x = min(point1[0], point2[0])
        min_y = min(point1[1], point2[1])
        rectan_width = abs(point1[0] - point2[0])
        rectan_height = abs(point1[1] - point2[1])
        img[0:min_y, min_x:img_width] = 0
        img[min_y:img_height, min_x + rectan_width:img_width] = 0
        img[min_y + rectan_height:img_height, 0:min_x + rectan_width] = 0
        img[0:min_y + rectan_height, 0:min_x] = 0
        cv2.imshow('image', img)
        U0[0:min_y, min_x:img_width] = 0
        U0[min_y:img_height, min_x + rectan_width:img_width] = 0
        U0[min_y + rectan_height:img_height, 0:min_x + rectan_width] = 0
        U0[0:min_y + rectan_height, 0:min_x] = 0


        # 滤出最中心的高亮像素块
        _, binary_image = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
        # binary_image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 1)
        binary_image_blurred = cv2.GaussianBlur(binary_image, (1, 1), 50)
        contours, _ = cv2.findContours(binary_image_blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        counter_x = []
        counter_y = []
        counter_w = []
        counter_h = []
        for contour in contours:
            # cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            counter_x.append(x)
            counter_y.append(y)
            counter_w.append(w)
            counter_h.append(h)
            # aspect_ratio = float(w) / h
            # chars.append(img[y:y + h, x:x + w])
        # 找到最大宽度的矩形并画出
        max_index = counter_w.index(max(counter_w))
        x_center = counter_x[max_index]
        y_center = counter_y[max_index]
        w_center = counter_w[max_index]
        h_center = counter_h[max_index]
        cv2.rectangle(img, (x_center, y_center), (x_center + w_center, y_center + h_center), (0, 255, 0), 8)

        cv2.namedWindow('Image with Bright Spots', 0)
        cv2.imshow("Image with Bright Spots", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 频谱位移到中心
        # delta_x = int(0.5*rectan_width+min_x-0.5*img_width)
        # delta_y = int(0.5*rectan_height+min_y-0.5*img_height)
        delta_x = int(0.5*w_center+x_center-0.5*img_width)
        delta_y = int(0.5*h_center+y_center-0.5*img_height)
        U0 = np.roll(U0, -delta_x, axis=1)
        U0 = np.roll(U0, -delta_y, axis=0)


# 原始图像
image0 = Image.open(r'F:\Data\20240717\0hunhe\yili_fangjie_lv\Image__2024-07-17__13-39-56.bmp')
width, height = image0.size
# 生成原始图像和背景图的灰度图和数组
grayscale_image = image0.convert("L")
grayscale_array = np.asarray(grayscale_image)

# FFT变换生成频谱图
U0 = fftshift(fft2(grayscale_array))
# 超出灰度阈值，降幂
U1 = np.log(1 + np.abs(U0))
plt.imsave('C:/Users/d1009/Desktop/test/FFT.jpg', U1, cmap="gray")  # 频谱图保存路径，需要先设定
img_paths = 'C:/Users/d1009/Desktop/test/FFT.jpg'
for img_path in glob.glob(img_paths):
    img_id = os.path.basename(img_path)
    # img_id = img_name.split('.')[0]
    cut(img_path)

# 对截取后的U0重建
U0_processed = ifft2(ifftshift(U0))
# 波长
lam = 532e-9
# 像素大小
pix = 0.223e-6
k = 2*np.pi/lam

# 重建距离
z1 = -0.00060
z2 = 0.00060
z_interval = 0.00005  # 间距

# 空间频率
fx = np.linspace(-1/(2*pix), 1/(2*pix), width)
fy = np.linspace(-1/(2*pix), 1/(2*pix), height)
FX, FY = np.meshgrid(fx, fy)
temp = 1 - ((lam*FX)**2 + (lam*FY)**2)
temp[temp < 0] = 0
z = np.linspace(z1, z2, int((z2-z1)/z_interval)+2)

# 开始重建
for i in range(len(z)):
    g = np.exp(1j * (2*np.pi/lam) * z[i] * np.sqrt(temp))
    g[temp < 0] = 0
    g = fftshift(g)
    U1 = fft2(fftshift(U0_processed))
    U2 = U1*g
    U3 = ifftshift(ifft2(U2))
    plt.imsave(r'C:/Users/d1009/Desktop/test/offaxis/result/offaxis_{:d}_{:.7f}.jpg'.format(i + 1, z[i]), abs(U3), cmap="gray")  # 重建后图像保存路径
