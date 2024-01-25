import numpy as np
from PIL import Image
from numpy.fft import fftshift, fft2, ifft2, ifftshift
import matplotlib.pyplot as plt
# 同轴平面波菲涅尔变换
image = Image.open('E:\pycharm_test\Fig\\test_coaxis\holo_fig1.bmp')
width, height = image.size
grayscale_image = image.convert("L")
grayscale_array = np.asarray(grayscale_image)
# grayscale_image.save('output_image.jpg')
# 波长
lam = 532e-9
# 像素大小
pix = 1.72e-6
k = 2*np.pi/lam
# 重建距离
z1 = 0.01
z2 = 0.02
z_interval = 0.0001
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
    plt.imsave('E:\pycharm_test\Fig\\test_coaxis\Fresnel\\re_{:d}_{:.4f}.jpg'.format(i+1, z[i]), abs(U3), cmap="gray")
