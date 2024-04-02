import cv2
import numpy as np
from numpy.fft import fftshift, fft2, ifft2, ifftshift

img = cv2.imread('F:/Data/20240329/DOF/Limestone/Limestone_DOF_12.bmp', 0)

# 中值滤波
img_mida = cv2.medianBlur(img, 201)
img_sub = cv2.addWeighted(img, -1, img_mida, 1, 0)

# ret, thresh = cv2.threshold(sub_mida, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# _, thresh = cv2.threshold(img_sub, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# thresh = cv2.adaptiveThreshold(img_sub, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
_, thresh = cv2.threshold(img_sub, 15, 255, cv2.THRESH_BINARY)

cv2.imwrite('F:/Data/20240329/temp/origin.jpg', img)
cv2.imwrite('F:/Data/20240329/temp/mida.jpg', img_mida)
cv2.imwrite('F:/Data/20240329/temp/sub.jpg', img_sub)
cv2.imwrite('F:/Data/20240329/temp/binary.jpg', thresh)

# 形态学
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11), (-1, -1))
img_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel1, (-1, -1))
cv2.imwrite('F:/Data/20240329/temp/open.jpg', img_open)

kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55, 55), (-1, -1))
img_close = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel2, (-1, -1))
cv2.imwrite('F:/Data/20240329/temp/close.jpg', img_close)

contours, _ = cv2.findContours(img_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
count = 0
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for contour in contours:
    if cv2.contourArea(contour) * 0.098 ** 2 > 100:
        count += 1
        cropped_img = None
        mask = None
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 5)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(img, '{:d}'.format(count), (x + 10, y + 10), cv2.FONT_HERSHEY_PLAIN, 5,
                    (0, 255, 0), 10, cv2.LINE_AA)

cv2.imwrite('F:/Data/20240329/temp/contour.jpg', img)



# # 顶帽运算
# # 设置卷积核
# kernel = np.ones((10, 10), np.uint8)
#
# # 图像顶帽运算
# result = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
# cv2.imwrite('F:/Data/20240329/temp/hat.jpg', result)
#
# _, thresh = cv2.threshold(result, 25, 255, cv2.THRESH_BINARY)
# cv2.imwrite('F:/Data/20240329/temp/hat_thresh.jpg', thresh)


