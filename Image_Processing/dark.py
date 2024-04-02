# 特定亮度以上的部分调暗
import os

import cv2
import numpy as np


def adjust_brightness(image_path, output_path, threshold=128):
    """
    调整指定图片中亮度阈值以上的部分，实现调暗效果。
    :param image_path: 原始图片路径
    :param output_path: 输出图片路径
    :param threshold: 亮度阈值
    """
    # 读取图片
    image = cv2.imread(image_path)

    # 将图片从BGR转换为HSV色彩空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 分离V通道（亮度）
    v_channel = hsv[:, :, 2]

    # 创建一个掩码，用于标记亮度高于阈值的像素
    mask = v_channel > threshold

    # 创建一个新的V通道，其中高于阈值的像素被设置为阈值
    v_channel_adjusted = np.where(mask, threshold, v_channel)

    # 合并调整后的V通道与原始的H和S通道
    hsv_adjusted = cv2.merge([hsv[:, :, 0], hsv[:, :, 1], v_channel_adjusted])

    # 将图片从HSV色彩空间转换回BGR
    image_adjusted = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)

    # 保存调整后的图片
    cv2.imwrite(output_path, image_adjusted)


if __name__ == '__main__':
    threshold_light = 128
    input_p = 'F:/Data/20240326/Gypsum'
    output_p = 'F:/Data/20240326/Gypsum_darken'
    images = os.listdir(input_p)
    image_files = [f for f in images if any(ext in f.lower() for ext in ('.jpg', '.jpeg', '.png', '.bmp'))]
    number = 0

    for image in image_files:
        number += 1
        adjust_brightness(input_p + '/' + image, output_p + '/' + str(number) + '.jpg', threshold_light)
