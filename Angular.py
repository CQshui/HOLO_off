import numpy as np
from numpy.fft import fftshift, fft2, ifft2, ifftshift


def Angular(lam, pix, z, image):
    width, height = image.size
    grayscale_image = image.convert("L")
    grayscale_array = np.asarray(grayscale_image)
    # 空间频率
    fx = np.linspace(-1/(2*pix), 1/(2*pix), width)
    fy = np.linspace(-1/(2*pix), 1/(2*pix), height)
    FX, FY = np.meshgrid(fx, fy)
    temp = 1 - ((lam*FX)**2 + (lam*FY)**2)
    # temp[temp < 0] = 0
    g = np.exp(1j * (2*np.pi/lam) * z * np.sqrt(temp))
    # g[temp < 0] = 0
    g = fftshift(g)
    U1 = fft2(fftshift(grayscale_array))
    U2 = U1*g
    U3 = ifftshift(ifft2(U2))

    return U3
