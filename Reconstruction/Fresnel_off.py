import glob
import os
import cv2
import numpy as np
from PIL import Image
from numpy.fft import fftshift, fft2, ifft2, ifftshift
import matplotlib.pyplot as plt
from phase_reconstruction import phase_unwrap
from scipy.signal.windows import tukey



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
        max_x = max(point1[0], point2[0])
        max_y = max(point1[1], point2[1])
        rectan_width = abs(point1[0] - point2[0])
        rectan_height = abs(point1[1] - point2[1])

        # 定义掩膜
        mask = np.zeros((img_height, img_width), dtype=np.float32)

        mask[min_y:min_y + rectan_height, min_x:min_x + rectan_width] = 1

        # 对示例图像和U0使用掩膜
        img_masked = img*mask
        img_masked = np.asarray(img_masked, dtype=np.uint8)
        cv2.imshow('image', img_masked)
        U0 = U0*mask

        # 滤出最中心的高亮像素块
        _, binary_image = cv2.threshold(img_masked, 180, 255, cv2.THRESH_BINARY)
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
        cv2.rectangle(img_masked, (x_center, y_center), (x_center + w_center, y_center + h_center), (0, 255, 0), 8)

        cv2.namedWindow('Image with Bright Spots', 0)
        cv2.imshow("Image with Bright Spots", img_masked)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 频谱位移到中心
        # delta_x = int(0.5*rectan_width+min_x-0.5*img_width)
        # delta_y = int(0.5*rectan_height+min_y-0.5*img_height)
        delta_x = int(0.5*w_center+x_center-0.5*img_width)
        delta_y = int(0.5*h_center+y_center-0.5*img_height)
        U0 = np.roll(U0, -delta_x, axis=1)
        U0 = np.roll(U0, -delta_y, axis=0)


# 参数设定
# 波长
lam = 532e-9
# 像素大小
pix = 0.223e-6
k = 2*np.pi/lam

# 重建距离
z1 = 0.000005
z2 = 0.000050
z_interval = 0.000005

# 原始图像
image0 = Image.open(r'F:\Data\20240717\0hunhe\yili_fangjie_lv\Image__2024-07-17__13-38-18.bmp')   # todo
width, height = image0.size

# 生成原始图像和背景图的灰度图和数组
grayscale_image = image0.convert("L")
grayscale_array = np.asarray(grayscale_image)

# 背景图，需要减背景可以加上
# back = Image.open('D:\\Desktop\\test\\bg\\Bg-noRule-Gyp.jpg')
# background_image = back.convert("L")
# background_array = np.asarray(background_image)
# 减背景
# grayscale_array = grayscale_array-0.5*background_array
# plt.imsave('D:\Desktop\\test\\back_removed.jpg', grayscale_array, cmap="gray")

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
plt.imsave('C:/Users/d1009/Desktop/test/U0.jpg', abs(U0_processed), cmap="gray")

# 是否使用tukey窗函数切趾
tukey_choice = False
if tukey_choice:
    # 创建一维Tukey窗口
    tukey_window_1d = tukey(min(height-100, width-100), alpha=0.05)

    # 通过外积将一维窗口转换为二维窗口
    tukey_window_2d = np.outer(tukey_window_1d, tukey_window_1d)

    # 调整窗口的大小以匹配图像的尺寸
    tukey_window_2d = cv2.resize(tukey_window_2d, (width-100, height-100))
    img_mask = np.zeros((height, width), dtype=np.float32)
    img_mask[50:height-50, 50:width-50] = tukey_window_2d

    U0_processed =  U0_processed*img_mask



x = np.linspace(-pix*width/2, pix*width/2, width)
y = np.linspace(-pix*height/2, pix*height/2, height)
x, y = np.meshgrid(x, y)
z = np.linspace(z1, z2, int((z2-z1)/z_interval)+2)

# 开始重建
for i in range(len(z)):
    r = np.sqrt(x**2+y**2+z[i]**2)
    # h= 1/ (1j*lam*z[i]) * np.exp(1j*k/ (2*z[i]) * (x**2+ y**2))
    h = z[i]/(1j*lam*r**2)*np.exp(1j*k*r)    # changed, h = 1/(1j*lam*r)*np.exp(1j*k*r)
    H = fft2(fftshift(h))*pix**2
    U1 = fft2(fftshift(U0_processed))
    U2 = U1*H
    U3 = ifftshift(ifft2(U2))
    U4 = phase_unwrap(U3)
    plt.imsave('C:/Users/d1009/Desktop/test/offaxis/result/offaxis_{:d}_{:.7f}.jpg'.format(i + 1, z[i]), abs(U3), cmap="gray")  # 重建后图像保存路径
    plt.imsave('C:/Users/d1009/Desktop/test/unwrap/unwrap_{:d}_{:.7f}.jpg'.format(i + 1, z[i]), U4, cmap="gray")
