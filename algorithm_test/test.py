import cv2
import numpy as np
from scipy import ndimage
kernel_3X3=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
kernel_5X5=np.array([[-1,-1,-1,-1,-1],[-1,1,2,1,-1],[-1,2,4,2,-1],[-1,1,2,1,-1],[-1,-1,-1,-1,-1]])

img=cv2.imread("1.jpg",0)
k3=ndimage.convolve(img,kernel_3X3)#卷积
k5=ndimage.convolve(img,kernel_5X5)#卷积

blurred=cv2.GaussianBlur(img,(11,11),0)
g_hpf=img-blurred
#两种高通滤波的效果
cv2.imshow("3x3",k3)
cv2.imshow("5x5",k5)
#通过对图像应用低通滤波器之后，与原始图像计算差值
cv2.imshow("g_hpf",g_hpf)
cv2.waitKey()
cv2.destroyAllWindows()