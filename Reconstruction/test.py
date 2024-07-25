import os
import cv2
import numpy as np
import pywt
from PIL import Image
from numpy.fft import fftshift, fft2, ifft2, ifftshift
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
from tqdm import tqdm

img_pth = 'C:/Users\d1009\Desktop\process/34.jpg'
save_pth = 'C:/Users\d1009\Desktop\process/34_2.png'
img = Image.open(img_pth)
width, height = img.size
gray_image = img.convert("L")
gray = np.asarray(gray_image)

# # FFT变换生成频谱图
# u0 = fftshift(fft2(gray))
# # 超出灰度阈值，降幂
# u1 = np.log(1 + np.abs(u0))
# fft_pth = os.path.join(save_pth, 'FFT.jpg')
# plt.imsave(fft_pth, u1, cmap="gray")


# 去背景示例
# bg = Image.open('F:/Data/20240503/bg.bmp')
# bg = np.asarray(bg.convert('L'))
#
# sub = img-0.5*bg
# sub = sub*255/(max(max(row) for row in sub)-(min(min(row) for row in sub)))
# plt.imsave('F:/Data/20240503/bg/sub.bmp', sub, cmap="gray")
#
# div = img/(10+bg)
# div = div*255/(max(max(row) for row in div)-(min(min(row) for row in div)))
# plt.imsave('F:/Data/20240503/bg/div.bmp', div, cmap="gray")


# # 小波示例
# wavelet = 'db2'
# coeff = pywt.wavedec2(gray[:, :], wavelet, level=1)
#
# c0 = coeff[0]
# c1 = coeff[1][0]
# c2 = coeff[1][1]
# c3 = coeff[1][2]
# plt.imsave('F:/Data/temp/wavelet/c0.bmp', c0, cmap="gray")
# plt.imsave('F:/Data/temp/wavelet/c1.bmp', c1, cmap="gray")
# plt.imsave('F:/Data/temp/wavelet/c2.bmp', c2, cmap="gray")
# plt.imsave('F:/Data/temp/wavelet/c3.bmp', c3, cmap="gray")

# 二值化

# # 全局
# ret, mask_all = cv2.threshold(src=gray,  # 要二值化的图片
#                               thresh=60,  # 全局阈值
#                               maxval=255,  # 大于全局阈值后设定的值
#                               type=cv2.THRESH_BINARY)  # 设定的二值化类型，

# 局部
mask_local = cv2.adaptiveThreshold(src=gray,  # 要进行处理的图片
                                   maxValue=255,  # 大于阈值后设定的值
                                   adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,  # 自适应方法
                                   thresholdType=cv2.THRESH_BINARY,  # 同全局阈值法中的参数一样
                                   blockSize=501,  # 方阵（区域）大小，
                                   C=30)  # 常数项，
#
#
mask_all = cv2.bitwise_not(mask_local)
cv2.imwrite(save_pth, mask_all)