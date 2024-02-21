import cv2
import numpy as np

# pix = 0.098e-6  # unit: m
pix = 0.098  # unit: um
img_path = 'D:\\Desktop\\test\\offaxis\\Gypsum\\test.jpg'
img = cv2.imread(img_path, 0)
im_height, im_width = img.shape
# _, binary_image = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
# ret2, binary_image = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
binary_image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 1505, 50)
inverted_image = cv2.bitwise_not(binary_image)
cv2.imwrite('D:\\Desktop\\test\\offaxis\\Gypsum\\result\\binary.jpg', inverted_image)
# 图片边缘平滑化，考虑15个像素
inverted_image[0:15, 0:im_width-15] = 0
inverted_image[0:im_height-15, im_width-15:im_width] = 0
inverted_image[im_height-15:im_height, 15:im_width] = 0
inverted_image[15:im_height, 0:15] = 0
# 开运算&闭运算
kernel1 = np.ones((15, 15), np.uint8)
opened = cv2.morphologyEx(inverted_image, cv2.MORPH_OPEN, kernel1)
kernel2 = np.ones((5, 5), np.uint8)
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel2)
# 转变为三通道图片
bgr_image = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)
cv2.imwrite('D:\\Desktop\\test\\offaxis\\Gypsum\\result\\closed.jpg', bgr_image)
rgb_image = bgr_image[:, :, ::-1]
# 计算连通域数量，框选计数
contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
count = 0
# 创建参数列表
A = 0
L = 0
area = []
perimeter = []
diameter = []
roundness = []
for contour in contours:
    count += 1
    # cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(bgr_image, (x, y), (x + w, y + h), (255, 0, 0), 5)
    cv2.putText(bgr_image, '{:d}'.format(count), (x+10, y+10), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 10, cv2.LINE_AA)
    area.append(cv2.contourArea(contour)*pix**2)
    perimeter.append(cv2.arcLength(contour, True)*pix)
    diameter.append(4*area[count-1]/perimeter[count-1])
    roundness.append(4*np.pi*area[count-1]/perimeter[count-1]**2)
print('count =', count)
print('area =', area)
print('perimeter =', perimeter)
print('diameter =', diameter)
print('roundness =', roundness)
cv2.imwrite('D:\\Desktop\\test\\offaxis\\Gypsum\\result\\numbered.jpg', bgr_image)

# cv2.namedWindow('TEST', 0)
# cv2.imshow("TEST", inverted_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
