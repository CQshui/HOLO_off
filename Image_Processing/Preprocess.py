import cv2
import numpy as np
from scipy import ndimage
# from Fractal import box_counting_dimension
# from Curvature import css
# from Blur import blur
from Hull import hull_solid, defect
from Curvature import css
from skimage.feature import graycomatrix, graycoprops


def calculate_third_moment(img, mean):
    color_sum = 0
    y, x = img.shape
    for i in range(y):
        for j in range(x):
            color_sum += (img[i, j] - mean) ** 3

    if color_sum >= 0:
        third_moment = (color_sum / (y * x)) ** (1/3)
    else:
        third_moment = -(-color_sum / (y * x)) ** (1/3)

    return third_moment


def calculate_glcm(img):
    glcm = graycomatrix(
        img, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], 256, symmetric=True, normed=True)
    mean = []
    for level1 in glcm:
        for level2 in level1:
            for i in level2:
                mean.append(np.mean(i))

    mean = np.array(mean)
    Glcm_new = mean.reshape((256, 256, 1, 1))

    result = []
    for prop in {'contrast', 'dissimilarity',
                 'homogeneity', 'energy', 'correlation', 'ASM'}:
        temp = graycoprops(Glcm_new, prop)
        result.append(temp)

    entropy = -np.sum(Glcm_new * np.log2(Glcm_new + np.finfo(float).eps))
    result.append(entropy)

    return result


def preprocess(pix, mask_path, image_path, file_id=0):
    # pix = 0.098e-6  # unit: m
    # pix = 0.098  # unit: um
    im_mask = cv2.imread(mask_path, 0)
    im_height, im_width = im_mask.shape

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

    # 创建参数列表
    number = []
    # 几何、形状参数
    area = []
    perimeter = []
    diameter = []
    roundness = []
    hull = []
    solid = []
    aspect_ratio = []
    depth_sum = []
    lenth = []
    width = []

    # 标签
    label = []
    # 灰度参数
    gray1 = []
    gray2 = []
    gray3 = []
    # 纹理参数
    correlation = []
    contrast = []
    asm = []
    entropy = []
    # Hu
    Hu1 = []
    Hu2 = []
    Hu3 = []
    Hu4 = []
    Hu5 = []
    Hu6 = []
    Hu7 = []

    for contour in contours:
        if cv2.contourArea(contour)*pix**2 > 50:
            count += 1
            cv2.drawContours(bgr_image, [contour], -1, (0, 255, 0), 5)
            x, y, w, h = cv2.boundingRect(contour)
            # cv2.rectangle(bgr_image, (x, y), (x + w, y + h), (255, 0, 0), 5)
            cv2.putText(bgr_image, '{:d}'.format(count), (x+10, y+10), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 10, cv2.LINE_AA)

            # id
            number.append(count)
            # 面积
            area.append(cv2.contourArea(contour)*pix**2)
            # 周长
            perimeter.append(cv2.arcLength(contour, True)*pix)
            # 等效直径
            diameter.append(4*area[count-1]/perimeter[count-1])
            # 圆形度
            roundness.append(4*np.pi*area[count-1]/perimeter[count-1]**2)
            # 凸缺陷、实心度、凸缺陷深度和
            hull_element, solid_element, depth_element = defect(contour, count)
            hull.append(hull_element)
            solid.append(solid_element)
            depth_sum.append(depth_element)
            # 标签
            label.append(0)     # todo 1_石膏，2_石灰石，0_未知

            # 拟合椭圆
            ellipse = cv2.fitEllipse(contour)
            (center, axes, orientation) = ellipse
            major_axis_length = max(axes)
            minor_axis_length = min(axes)
            # 长短轴比
            aspect_ratio.append(major_axis_length / minor_axis_length)
            # 长轴
            lenth.append(major_axis_length)
            width.append(minor_axis_length)

            # 灰度特征
            edge = 25
            gray_img = cv2.imread(image_path, 0)
            cropped_img = gray_img[y - edge:y + h + edge,
                          x - edge:x + w + edge]
            # 一阶矩
            gray_mean = np.mean(cropped_img)
            gray1.append(gray_mean)
            # 二阶矩
            gray_std = np.std(cropped_img)
            gray2.append(gray_std)
            # 三阶矩
            gray_S = calculate_third_moment(cropped_img, gray_mean)
            gray3.append(gray_S)

            # 纹理特征（灰度共生矩阵）
            result = calculate_glcm(cropped_img)
            # 'contrast', 'dissimilarity',
            # 'homogeneity', 'energy', 'correlation', 'ASM'
            contrast.append(result[0][0][0])
            asm.append(result[5][0][0])
            correlation.append(result[4][0][0])
            entropy.append(result[6])

            # Hu矩
            M = cv2.moments(contour)
            Hm = cv2.HuMoments(M)
            h1, h2, h3, h4, h5, h6, h7 = Hm
            Hu1.append(h1[0])
            Hu2.append(h2[0])
            Hu3.append(h3[0])
            Hu4.append(h4[0])
            Hu5.append(h5[0])
            Hu6.append(h6[0])
            Hu7.append(h7[0])



    if file_id == 1:
        area.insert(0, 'area')
        perimeter.insert(0, 'perimeter')
        diameter.insert(0, 'diameter')
        roundness.insert(0, 'roundness')
        number.insert(0, 'number')
        label.insert(0, 'label')
        hull.insert(0, 'hull')
        solid.insert(0, 'solid')
        aspect_ratio.insert(0, 'aspect_ratio')
        depth_sum.insert(0, 'depth_sum')
        lenth.insert(0, 'lenth')
        width.insert(0, 'width')
        gray1.insert(0, 'gray1')
        gray2.insert(0, 'gray2')
        gray3.insert(0, 'gray3')
        contrast.insert(0, 'contrast')
        asm.insert(0, 'ASM')
        correlation.insert(0, 'correlation')
        entropy.insert(0, 'entropy')
        Hu1.insert(0, 'Hu1')
        Hu2.insert(0, 'Hu2')
        Hu3.insert(0, 'Hu3')
        Hu4.insert(0, 'Hu4')
        Hu5.insert(0, 'Hu5')
        Hu6.insert(0, 'Hu6')
        Hu7.insert(0, 'Hu7')


    # cv2.imwrite('C:\\Users\\d1009\\Desktop\\test\\output\\numbered.jpg', bgr_image)
    return number, area, width, lenth, aspect_ratio, roundness, solid, gray1, gray2, gray3, contrast, asm, correlation, entropy, Hu1, Hu2, Hu3, Hu4, Hu5, Hu6, Hu7, label


if __name__ == '__main__':
    result = preprocess(pix=0.223,
                        mask_path='K:/pythonProject/Torch_test/result/35.png',
                        image_path='K:/pythonProject/Torch_test/datas/test/predict/35.png'
                        )

