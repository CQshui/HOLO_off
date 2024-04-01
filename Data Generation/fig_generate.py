import os

import cv2
import numpy as np


def generate(img_path, save_path, choice, types, file_number=0):
    pix = 0.098
    # img_path = 'C:\\Users\\d1009\\Desktop\\generation\\input\\Gypsum\\DOF_gypsum.bmp'
    img = cv2.imread(img_path, 0)
    print(img_path)
    im_height, im_width = img.shape
    binary_image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                         805, 5)
    inverted_image = cv2.bitwise_not(binary_image)
    # 图片边缘规整，考虑edge个像素
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
    # 画出轮廓图
    contour_img = np.zeros_like(img)
    contour_img = cv2.cvtColor(contour_img, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        if cv2.contourArea(contour) * pix ** 2 > 100:
            count += 1
            cropped_img = None
            mask = None
            cv2.drawContours(contour_img, [contour], -1, (255, 255, 255), 5)
            x, y, w, h = cv2.boundingRect(contour)

            # 在边缘留出空隙edge_generation
            edge_generation = 25
            # 裁剪并保存
            if choice == 'origin':
                cropped_img = img[y - edge_generation:y + h + edge_generation,
                              x - edge_generation:x + w + edge_generation]

            elif choice == 'binary':
                cropped_img = bgr_image[y - edge_generation:y + h + edge_generation,
                              x - edge_generation:x + w + edge_generation]

            elif choice == 'contour':
                cropped_img = contour_img[y - edge_generation:y + h + edge_generation,
                              x - edge_generation:x + w + edge_generation]

            elif choice == 'mask':
                mask = np.zeros(img.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, (255, 255, 255), cv2.FILLED)
                to_crop_img = cv2.bitwise_and(img, img, mask=mask)
                cropped_img = to_crop_img[y - edge_generation:y + h + edge_generation,
                              x - edge_generation:x + w + edge_generation]

            cv2.imwrite(save_path + types + '/{:d}_{:d}.bmp'.format(file_number, count), cropped_img)

        # cv2.imwrite('C:\\Users\\d1009\\Desktop\\generation\\output\\Gypsum\\numbered.bmp', bgr_image)


if __name__ == '__main__':
    im_path = 'F:/20240326/Limestone'
    saving_path = 'F:/20240326/Single/'
    images = os.listdir(im_path)
    image_files = [f for f in images if any(ext in f.lower() for ext in ('.jpg', '.jpeg', '.png', '.bmp'))]
    number = 0
    for image in image_files:
        im_path = 'F:/20240326/Limestone'
        number += 1
        im_path = im_path + '/' + image
        generate(im_path, saving_path, 'origin', 'Limestone', number)