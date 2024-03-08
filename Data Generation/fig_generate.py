import cv2
import numpy as np

pix = 0.098
img_path = 'C:\\Users\\d1009\\Desktop\\generation\\input\\Gypsum\\DOF_gypsum.bmp'
img = cv2.imread(img_path, 0)
im_height, im_width = img.shape
binary_image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                     2005, 20)
inverted_image = cv2.bitwise_not(binary_image)
# 图片边缘平滑化，考虑edge个像素
edge = 50
inverted_image[0:edge, 0:im_width-edge] = 0
inverted_image[0:im_height-edge, im_width-edge:im_width] = 0
inverted_image[im_height-edge:im_height, edge:im_width] = 0
inverted_image[edge:im_height, 0:edge] = 0
# 开运算&闭运算
kernel1 = np.ones((5, 5), np.uint8)
opened = cv2.morphologyEx(inverted_image, cv2.MORPH_OPEN, kernel1)
kernel2 = np.ones((15, 15), np.uint8)
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel2)
# 转变为三通道图片
bgr_image = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)
rgb_image = bgr_image[:, :, ::-1]
# 计算连通域数量，框选计数
contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
count = 0

for contour in contours:
    if cv2.contourArea(contour) * pix ** 2 > 100:
        count += 1
        cv2.drawContours(bgr_image, [contour], -1, (0, 255, 0), 5)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(bgr_image, '{:d}'.format(count), (x + 10, y + 10), cv2.FONT_HERSHEY_PLAIN,
                    5, (0, 255, 0), 10,cv2.LINE_AA)
        # 在边缘留出空隙edge_generation
        edge_generation = 25
        # 裁剪图片
        cropped_img = img[y-edge_generation:y+h+edge_generation, x-edge_generation:x+w+edge_generation]
        # 保存裁剪后的图片
        cv2.imwrite('C:\\Users\\d1009\\Desktop\\generation\\output\\Gypsum\\crop_{:d}.bmp'.format(count),
                    cropped_img)

    cv2.imwrite('C:\\Users\\d1009\\Desktop\\generation\\output\\Gypsum\\numbered.bmp', bgr_image)

