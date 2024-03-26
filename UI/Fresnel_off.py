import glob
import os
import cv2
import numpy as np
from PIL import Image
from numpy.fft import fftshift, fft2, ifft2, ifftshift
import matplotlib.pyplot as plt
from phase_reconstruction import phase_unwrap

img = None
img_height = None
img_width = None
U0 = None
width = None
height = None


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
    global point1, point2, U0, height, width
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
        U0[min_y:img_height, min_x + width:img_width] = 0
        U0[min_y + height:img_height, 0:min_x + width] = 0
        U0[0:min_y + height, 0:min_x] = 0

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


def Fresnel_re(path, lam, pix, z1, z2, z_interval, name='Default'):
    global U0, width, height
    # 背景图
    # back = Image.open('D:\\Desktop\\test\\bg\\Bg-noRule-Gyp.jpg')
    # 原始图像
    image0 = Image.open(path)
    width, height = image0.size

    # 生成原始图像和背景图的灰度图和数组
    grayscale_image = image0.convert("L")
    grayscale_array = np.asarray(grayscale_image)
    # background_image = back.convert("L")
    # background_array = np.asarray(background_image)
    # 减背景
    # grayscale_array = grayscale_array-0.5*background_array
    # plt.imsave('D:\Desktop\\test\\back_removed.jpg', grayscale_array, cmap="gray")

    # FFT变换生成频谱图
    U0 = fftshift(fft2(grayscale_array))
    # 超出灰度阈值，降幂
    U1 = np.log(1 + np.abs(U0))
    plt.imsave('C:\\Users\\d1009\\Desktop\\test\\FFT.jpg', U1, cmap="gray")
    img_paths = 'C:\\Users\\d1009\\Desktop\\test\\FFT.jpg'
    for img_path in glob.glob(img_paths):
        img_id = os.path.basename(img_path)
        # img_id = img_name.split('.')[0]
        cut(img_path)

    # 对截取后的U0重建
    U0_processed = ifft2(ifftshift(U0))
    k = 2*np.pi/lam
    x = np.linspace(-pix*width/2, pix*width/2, width)
    y = np.linspace(-pix*height/2, pix*height/2, height)
    x, y = np.meshgrid(x, y)
    z = np.linspace(z1, z2, int((z2-z1)/z_interval)+1)
    for i in range(len(z)):
        r = np.sqrt(x**2+y**2+z[i]**2)
        # h= 1/ (1j*lam*z[i]) * np.exp(1j*k/ (2*z[i]) * (x**2+ y**2))
        h = z[i]/(1j*lam*r**2)*np.exp(1j*k*r)    # changed, h = 1/(1j*lam*r)*np.exp(1j*k*r)
        H = fft2(fftshift(h))*pix**2
        U1 = fft2(fftshift(U0_processed))
        U2 = U1*H
        U3 = ifftshift(ifft2(U2))
        U4 = phase_unwrap(U3)
        # new_U4 = [[127.5+x/2 for x in row] for row in U4]
        off_axis_dir = 'C:\\Users\\d1009\\Desktop\\test\\offaxis\\result\\{:}'.format(name)
        off_img_path = off_axis_dir + '\\offaxis_{:d}_{:.7f}.jpg'.format(i + 1, z[i])
        if not os.path.exists(off_axis_dir):
            os.makedirs(off_axis_dir)
        plt.imsave(off_img_path, abs(U3), cmap="gray")
        plt.imsave('C:\\Users\\d1009\\Desktop\\test\\unwrap\\unwrap_{:d}_{:.7f}.jpg'.format(i + 1, z[i]), U4, cmap="gray")
        # plt.imsave('D:\\Desktop\\test\\unwrap\\phase_correct_{:d}_{:.7f}.jpg'.format(i + 1, z[i]), abs(U5), cmap="gray")
