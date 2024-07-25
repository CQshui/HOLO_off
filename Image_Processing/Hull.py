import cv2
import matplotlib.pyplot as plt
import numpy as np


def hull_solid(contour, count):
    pix = 0.098
    hull = cv2.convexHull(contour)
    # 绘制原始点
    plt.plot(contour[:, 0, 0]*pix, contour[:, 0, 1]*pix, 'b.', label='Original Contour')

    # 绘制凸包
    hull_points = hull[:, 0, :]  # OpenCV返回的是嵌套数组，需要转换一下
    plt.plot(hull_points[:, 0]*pix, hull_points[:, 1]*pix, 'r-', label='Convex Hull')

    # 设置图例和显示图形
    plt.axis('equal')  # 设置坐标轴比例相等，使图形不扭曲
    plt.title('Gypsum {:d}'.format(count))
    plt.xlabel('X / μm')
    plt.ylabel('Y / μm')
    plt.legend(loc='upper right')
    plt.show()

    return None


def defect(contour, count):
    pix = 0.098
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)
    num_large_defects = 0
    depth_sum = 0
    large_defects = []

    # depth_threshold = 100*4*cv2.contourArea(contour)/cv2.arcLength(contour, True)
    # depth_threshold = 6000
    if defects is not None:
        large_defects = [d for d in defects
                         if d[0][3]/(256*(4*cv2.contourArea(contour)/cv2.arcLength(contour, True))) > 80/256]
        num_large_defects = len(large_defects)  # 大凸起瑕疵的个数
        if num_large_defects != 0:
            depth_sum += [depth[0][3] for depth in large_defects][0]
    # print(defects)

    # 计算实心度
    hull1 = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull1)
    area = cv2.contourArea(contour)
    solid = area/hull_area

    # 绘制原始点
    # plt.plot(contour[:, 0, 0]*pix, contour[:, 0, 1]*pix, 'b.', label='Original Contour')

    # 绘制凸包
    dot_x = []
    dot_y = []
    for i in hull:
        dot = contour[i[0]][0]
        dot_x.append(dot[0])
        dot_y.append(dot[1])

    # plt.plot(np.array(dot_x)*pix, np.array(dot_y)*pix, 'g-', label='Convex Hull')

    # 如果有凸缺陷，绘制
    far_x = []
    far_y = []
    if len(large_defects) > 0:
        for i in large_defects:
            f = i[0][2]
            far = contour[f][0]
            far_x.append(far[0])
            far_y.append(far[1])

        # plt.plot(np.array(far_x)*pix, np.array(far_y)*pix, 'ro', label='Convex Defect')  # 表示凸缺陷的远点

    # # 设置图例和显示图形
    # plt.axis('equal')  # 设置坐标轴比例相等，使图形不扭曲
    # plt.title('Gypsum {:d}'.format(count))
    # plt.xlabel('X / μm')
    # plt.ylabel('Y / μm')
    # plt.legend(loc='upper right')
    # plt.show()

    return num_large_defects, solid, depth_sum
