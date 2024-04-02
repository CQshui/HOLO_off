import cv2
import numpy as np
import os


# 椒盐噪声
def SaltAndPepper(src, percentage):
    SP_NoiseImg = src.copy()
    SP_NoiseNum = int(percentage * src.shape[0] * src.shape[1])
    for i in range(SP_NoiseNum):
        randR = np.random.randint(0, src.shape[0] - 1)
        randG = np.random.randint(0, src.shape[1] - 1)
        randB = np.random.randint(0, 3)
        if np.random.randint(0, 1) == 0:
            SP_NoiseImg[randR, randG, randB] = 0
        else:
            SP_NoiseImg[randR, randG, randB] = 255
    return SP_NoiseImg


# 高斯噪声
def addGaussianNoise(image, percentage):
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum = int(percentage * image.shape[0] * image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0, h)
        temp_y = np.random.randint(0, w)
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_Noiseimg


# 昏暗
def darker(image, percentage=0.9):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = int(image[xj, xi, 0] * percentage)
            image_copy[xj, xi, 1] = int(image[xj, xi, 1] * percentage)
            image_copy[xj, xi, 2] = int(image[xj, xi, 2] * percentage)
    return image_copy


# 亮度
def brighter(image, percentage=1.5):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = np.clip(int(image[xj, xi, 0] * percentage), a_max=255, a_min=0)
            image_copy[xj, xi, 1] = np.clip(int(image[xj, xi, 1] * percentage), a_max=255, a_min=0)
            image_copy[xj, xi, 2] = np.clip(int(image[xj, xi, 2] * percentage), a_max=255, a_min=0)
    return image_copy


# 旋转
def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, m, (w, h))
    return rotated


# 翻转
def flip(image):
    flipped_image = np.fliplr(image)
    return flipped_image


# 图片文件夹路径
input_dir = r'F:/Data/20240329/Single/Limestone/'
save_dir = r'I:/2023_pytorch110_classification_42-master/data/particle1/Limestone/'
for img_name in os.listdir(input_dir):
    img_path = input_dir + img_name
    img = cv2.imread(img_path)
    cv2.imwrite(save_dir + img_name[0:-4] + '.bmp', img)
    # 旋转
    rotated_90 = np.rot90(img, 1)
    cv2.imwrite(save_dir + img_name[0:-4] + '_r90.bmp', rotated_90)
