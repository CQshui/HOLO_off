import cv2
import numpy as np
# from Fractal import box_counting_dimension
# from Curvature import css
# from Blur import blur
from Hull import hull_solid, defect
from Curvature import css


def preprocess(pix, mask_path, image_path=None):
    # pix = 0.098e-6  # unit: m
    # pix = 0.098  # unit: um
    im_mask = cv2.imread(mask_path, 0)
    im_height, im_width = im_mask.shape

    if image_path != None:
        # 中值滤波
        img = cv2.imread(image_path, 0)
        img_mida = cv2.medianBlur(img, 201)
        img_sub = cv2.addWeighted(img, -1, img_mida, 1, 0)
        _, thresh = cv2.threshold(img_sub, 15, 255, cv2.THRESH_BINARY)
        inverted_image = thresh.copy()
        # 图片边缘规整，考虑edge个像素
        edge = 50
        inverted_image[0:edge, 0:im_width - edge] = 0
        inverted_image[0:im_height - edge, im_width - edge:im_width] = 0
        inverted_image[im_height - edge:im_height, edge:im_width] = 0
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
    # 创建参数列表
    number = []
    area = []
    perimeter = []
    diameter = []
    roundness = []
    roughness = []
    hull = []
    solid = []
    aspect_ratio = []
    depth_sum = []
    label = []

    for contour in contours:
        if cv2.contourArea(contour)*pix**2 > 100:
            count += 1
            cv2.drawContours(bgr_image, [contour], -1, (0, 255, 0), 5)
            x, y, w, h = cv2.boundingRect(contour)
            # cv2.rectangle(bgr_image, (x, y), (x + w, y + h), (255, 0, 0), 5)
            cv2.putText(bgr_image, '{:d}'.format(count), (x+10, y+10), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 10, cv2.LINE_AA)

            area.append(cv2.contourArea(contour)*pix**2)
            perimeter.append(cv2.arcLength(contour, True)*pix)
            diameter.append(4*area[count-1]/perimeter[count-1])
            roundness.append(4*np.pi*area[count-1]/perimeter[count-1]**2)
            number.append(count)
            # Roughness.append(linear_regression(contour))
            hull_element, solid_element, depth_element = defect(contour, count)
            hull.append(hull_element)
            solid.append(solid_element)
            depth_sum.append(depth_element)
            label.append(1)

            # 拟合椭圆
            ellipse = cv2.fitEllipse(contour)
            (center, axes, orientation) = ellipse
            major_axis_length = max(axes)
            minor_axis_length = min(axes)
            # 计算长短轴比
            aspect_ratio.append(major_axis_length / minor_axis_length)

            # 导数方向
            # if True:
            #     css(contour)

            # 画出凸包
            # if True:
            #     hull_solid(contour, count)

    area.insert(0, 'area')
    perimeter.insert(0, 'perimeter')
    diameter.insert(0, 'diameter')
    roundness.insert(0, 'roundness')
    number.insert(0, 'number')
    label.insert(0, 'label')
    # Roughness.insert(0, 'Roughness')
    # diffValue.insert(0, 'diffValue')
    # Fractal_Dim.insert(0, 'Fd')
    # gradient.insert(0, 'gradient')
    hull.insert(0, 'hull')
    solid.insert(0, 'solid')
    aspect_ratio.insert(0, 'aspect_ratio')
    depth_sum.insert(0, 'depth_sum')

    cv2.imwrite('C:\\Users\\d1009\\Desktop\\test\\output\\numbered.jpg', bgr_image)
    return number, area, perimeter, diameter, roundness, aspect_ratio, solid, label
