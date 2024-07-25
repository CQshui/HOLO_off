import os
import cv2
import numpy as np


def generate(img_path, save_path, choice, types, file_number=0, binary_path=None):
    pix = 0.223
    img = cv2.imread(img_path, 0)
    im_mask = cv2.imread(binary_path, 0)
    print(img_path)
    im_height, im_width = img.shape
    # 如果没有事先获取mask
    has_mask = False
    if has_mask:
        # 中值滤波
        img_mida = cv2.medianBlur(img, 201)
        img_sub = cv2.addWeighted(img, -1, img_mida, 1, 0)
        _, thresh = cv2.threshold(img_sub, 15, 255, cv2.THRESH_BINARY)
        inverted_image = thresh.copy()
        # 图片边缘规整，考虑edge个像素
        edge = 50
        inverted_image[0:edge, 0:im_width-edge] = 0
        inverted_image[0:im_height-edge, im_width-edge:im_width] = 0
        inverted_image[im_height-edge:im_height, edge:im_width] = 0
        inverted_image[edge:im_height, 0:edge] = 0
        # 开运算&闭运算
        kernel1 = np.ones((5, 5), np.uint8)
        opened = cv2.morphologyEx(inverted_image, cv2.MORPH_OPEN, kernel1)
        kernel2 = np.ones((11, 11), np.uint8)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel2)
        # 转变为三通道图片
        bgr_image = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
        # 计算连通域数量，框选计数
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        count = 0
        # 画出轮廓图
        contour_img = np.zeros_like(img)
        contour_img = cv2.cvtColor(contour_img, cv2.COLOR_GRAY2BGR)
    else:
        # 图片边缘规整
        edge = 50
        im_mask[0:edge, 0:im_width - edge] = 0
        im_mask[0:im_height - edge, im_width - edge:im_width] = 0
        im_mask[im_height - edge:im_height, edge:im_width] = 0
        im_mask[edge:im_height, 0:edge] = 0
        # 转变为三通道图片
        bgr_image = cv2.cvtColor(im_mask, cv2.COLOR_GRAY2BGR)
        # 计算连通域数量，框选计数
        contours, _ = cv2.findContours(im_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        count = 0
        # 画出轮廓图
        contour_img = np.zeros_like(img)
        contour_img = cv2.cvtColor(contour_img, cv2.COLOR_GRAY2BGR)

    # 对于每个轮廓
    for contour in contours:
        if cv2.contourArea(contour) * pix ** 2 > 50:
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

            elif choice == 'mask_binary':
                mask = np.zeros(img.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, (255, 255, 255), cv2.FILLED)
                to_crop_img = cv2.bitwise_and(bgr_image, bgr_image, mask=mask)
                cropped_img = to_crop_img[y - edge_generation:y + h + edge_generation,
                              x - edge_generation:x + w + edge_generation]

            save_type_path = os.path.join(save_path, types)
            if not os.path.exists(save_type_path):
                os.makedirs(save_type_path)
            cv2.imwrite(save_type_path + '/{:d}_{:d}.bmp'.format(file_number, count), cropped_img)

        # cv2.imwrite('C:\\Users\\d1009\\Desktop\\generation\\output\\Gypsum\\numbered.bmp', bgr_image)


if __name__ == '__main__':

    im_path = r'F:\Data\20240717\0hunhe\yili_fangjie_lv\select'
    saving_path = r'F:\Data\20240717\0hunhe\yili_fangjie_lv\single'
    mask_path = r'F:\Data\20240717\0hunhe\yili_fangjie_lv\binary'
    images = os.listdir(im_path)
    image_files = [f for f in images if any(ext in f.lower() for ext in ('.jpg', '.jpeg', '.png', '.bmp'))]
    number = 0
    for image in image_files:
        number += 1
        img_pth = im_path + '/' + image
        mask_pth = mask_path + '/' + image
        try:
            generate(img_pth, saving_path, 'origin', 'albite', number, mask_pth)

        except Exception as e:
            print(e)
            pass
