import numpy as np
from PIL import Image
from numpy.fft import fftshift, fft2, ifft2, ifftshift
import matplotlib.pyplot as plt
# 同轴平面波角谱重建
image = Image.open('D:\Desktop\\test\\eb18127ba65a9fa711227cc25c129ce.bmp')
width, height = image.size
grayscale_image = image.convert("L")
grayscale_array = np.asarray(grayscale_image)
# grayscale_image.save('output_image.jpg')
# 波长
lam = 532e-9
# 像素大小
pix = 1.72e-6
L = pix*width
H = pix*height
# 重建距离
z1 = 0.0100
z2 = 0.0120
z_interval = 0.0001
# 空间频率
fx = np.linspace(-1/(2*pix), 1/(2*pix), width)
fy = np.linspace(-1/(2*pix), 1/(2*pix), height)
FX, FY = np.meshgrid(fx, fy)
temp = 1 - ((lam*FX)**2 + (lam*FY)**2)
temp[temp < 0] = 0
z = np.linspace(z1, z2, int((z2-z1)/z_interval)+2)

for i in range(len(z)):
    g = np.exp(1j * (2*np.pi/lam) * z[i] * np.sqrt(temp))
    g[temp < 0] = 0
    g = fftshift(g)
    U1 = fft2(fftshift(grayscale_array))
    U2 = U1*g
    U3 = ifftshift(ifft2(U2))
    plt.imsave('D:\Desktop\\test\\re_{:d}_{:.4f}.jpg'.format(i+1, z[i]), abs(U3), cmap="gray")
