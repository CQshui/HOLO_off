import numpy as np
from PIL import Image
from numpy.fft import fftshift, fft2, ifft2, ifftshift
import matplotlib.pyplot as plt
import os


def fresnel(in_path, out_path):
    # 同轴平面波菲涅尔变换
    image = Image.open(in_path)
    bg = Image.open(bg_path)

    width, height = image.size
    grayscale_image = image.convert("L")
    grayscale_array = np.asarray(grayscale_image)
    bg_gray = bg.convert("L")
    bg_gray_array = np.asarray(bg_gray)
    # 减背景
    # grayscale_array = grayscale_array/bg_gray_array

    k = 2*np.pi/lam
    #
    x = np.linspace(-pix*width/2, pix*width/2, width)
    y = np.linspace(-pix*height/2, pix*height/2, height)
    x, y = np.meshgrid(x, y)
    z = np.linspace(z1, z2, int((z2-z1)/z_interval)+2)

    for i in range(len(z)):
        r = np.sqrt(x**2+y**2+z[i]**2)
        h = 1/(1j*lam*r)*np.exp(1j*k*r)
        H = fft2(fftshift(h))*pix**2
        U1 = fft2(fftshift(grayscale_array))
        U2 = U1*H
        U3 = ifftshift(ifft2(U2))

        # plt.imsave('F:/Result/re_{:d}_{:}.jpg'.format(i+1, z[i]), U3, cmap="gray")
        img_pth = out_path + '/' + 're_{:d}_{:}.jpg'.format(i+1, z[i])
        plt.imsave(img_pth, abs(U3), cmap='gray')


def batch(input_pth, output_pth):
    file_names = os.listdir(input_pth)
    image_files = [f for f in file_names if any(ext in f.lower() for ext in ('.jpg', '.jpeg', '.png', '.bmp'))]
    for image_file in image_files:
        image_path = os.path.join(input_pth, image_file)

        new_path = os.path.join(output_pth, image_file)
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        fresnel(image_path, new_path)


if __name__ == '__main__':
    # 波长
    lam = 532e-9
    # 像素大小
    pix = 0.098e-6
    # 重建距离
    z1 = 0.024e-2
    z2 = 0.028e-2
    z_interval = 0.0004e-2

    input_path = 'F:/Data/20240329/Gypsum'
    output_path = 'F:/Data/20240329/Reconstruction/Gypsum'
    bg_path = 'F:/Data/20240329/bg.bmp'
    batch(input_path, output_path)

