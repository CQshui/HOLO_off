import cv2
import numpy as np


img_path = 'D:\\Desktop\\test\\offaxis\\Iron\\test.jpg'
img = cv2.imread(img_path, 0)
im_height, im_width = img.shape
_, binary_image = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
inverted_image = cv2.bitwise_not(binary_image)
print(img.shape)
# 图片边缘平滑化，考虑15个像素
inverted_image[0:15, 0:im_width-15] = 0
inverted_image[0:im_height-15, im_width-15:im_width] = 0
inverted_image[im_height-15:im_height, 15:im_width] = 0
inverted_image[15:im_height, 0:15] = 0
# 开运算&闭运算
kernel1 = np.ones((10, 10), np.uint8)
opened = cv2.morphologyEx(inverted_image, cv2.MORPH_OPEN, kernel1)
kernel2 = np.ones((8, 8), np.uint8)
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel2)
# 转变为三通道图片
bgr_image = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
cv2.imwrite('D:\\Desktop\\test\\offaxis\\Iron\\result\\closed.jpg', bgr_image)
rgb_image = bgr_image[:, :, ::-1]
# 计算连通域数量，框选计数
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
count = 1
for contour in contours:
    # cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(bgr_image, (x, y), (x + w, y + h), (255, 0, 0), 5)
    cv2.putText(bgr_image, '{:d}'.format(count), (x+10, y+10), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 10, cv2.LINE_AA)
    count += 1
print(count-1)
cv2.imwrite('D:\\Desktop\\test\\offaxis\\Iron\\result\\numbered.jpg', bgr_image)

# cv2.namedWindow('TEST', 0)
# cv2.imshow("TEST", inverted_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
