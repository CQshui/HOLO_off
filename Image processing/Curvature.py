import numpy as np
import cv2
from scipy.interpolate import BSpline, make_interp_spline
import matplotlib.pyplot as plt


def curvature(contour):
    # print(contour)
    contour = np.array(contour).squeeze()
    # print(contour)
    if not (contour[0] == contour[-1]).all():
        contour = np.append(contour, [contour[0]], axis=0)
    # print(contour[0], contour[1], contour[2])
    # 计算一阶和二阶导，并计算曲率
    dx_dt = np.gradient(contour, axis=0)
    # print(dx_dt[0], dx_dt[1], dx_dt[2])
    dx_dt_norm = np.sqrt(dx_dt[:, 0] ** 2 + dx_dt[:, 1] ** 2)   # 模
    # print(dx_dt_norm)
    velocity = dx_dt / dx_dt_norm[:, None]  # 沿曲线的单位切向量

    d2x_dt2 = np.gradient(velocity, axis=0)
    dt_ds = 1 / dx_dt_norm
    cur = d2x_dt2 / dt_ds[:, None]  # 用二阶导数的模代表曲率
    # 通过曲率标准差表征曲线粗糙度
    roughness1 = np.std(cur[:, 0])
    roughness2 = np.std(cur[:, 1])
    roughness = np.sqrt(roughness1**2+roughness2**2)

    return roughness


def css(contour):
    x = contour[:, 0, 0]
    y = contour[:, 0, 1]
    # 相隔n个值做一次相减
    n = 60
    dx1 = []
    dy1 = []
    for i in range(len(x)):
        dx1.append(x[i]-x[i-n])
        dy1.append(y[i]-y[i-n])

    # 转为数组
    dx1 = np.array(dx1)
    dy1 = np.array(dy1)
    # 模
    norm = np.sqrt(dx1**2+dy1**2)
    # 单位切向量
    velocity_x = dx1 / norm
    velocity_y = dy1 / norm
    dot = np.arange(len(velocity_x))
    # 绘制线图
    plt.plot(dot, velocity_y, 'black')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    return None


def css1(contour):
    # 计算轮廓上每个点的曲率
    curvatures = []
    for i in range(len(contour) - 2):
        pt1 = contour[i]
        pt2 = contour[i + 1]
        pt3 = contour[i + 2]

        # 计算三个连续点形成的向量的叉积
        cross_product = np.cross(pt2 - pt1, pt3 - pt2)

        # 计算曲率
        curvature2 = np.linalg.norm(cross_product) / (np.linalg.norm(pt2 - pt1) * np.linalg.norm(pt3 - pt2))
        curvatures.append(curvature2)

    # 在不同尺度的高斯平滑下观察曲率的变化
    scales = np.linspace(1, 10, 10)  # 选择平滑尺度
    css_descriptors = []

    for scale in scales:
        # 对曲率进行高斯平滑
        ksize = 10
        # gaussian_kernel = cv2.getGaussianKernel(ksize, scale)
        gaussian_kernel = np.exp(-(np.arange(ksize) - ksize // 2) ** 2 / (2 * scale ** 2))
        gaussian_kernel /= np.sum(gaussian_kernel)
        smoothed_curvatures = np.convolve(curvatures, gaussian_kernel, mode='same')

        # 将平滑后的曲率作为CSS描述符保存
        css_descriptors.append(smoothed_curvatures)

    compare = []
    # 输出结果
    for i, descriptor in enumerate(css_descriptors[1:]):
        euclidean_distance = np.linalg.norm(css_descriptors[0] - descriptor)
        compare.append(euclidean_distance)
        # print(f"Scale {scales[i]}: {descriptor}")

    rough = np.std(compare)

    return rough
