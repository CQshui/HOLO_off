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
    # 一阶导，求相邻差值，diff会导致少一个值，用x[0]-x[-1]补足
    dfx1 = np.diff(x)
    dfx1 = np.append(dfx1, x[0]-x[-1])
    dfy1 = np.diff(y)
    dfy1 = np.append(dfy1, y[0]-y[-1])
    # 二阶导
    dfx2 = np.diff(dfx1)
    dfx2 = np.append(dfx2, dfx1[0]-dfx1[-1])
    dfy2 = np.diff(dfy1)
    dfy2 = np.append(dfy2, dfy1[0]-dfy1[-1])
    # 计算曲率
    K = np.abs((dfx1*dfy2-dfx2*dfy1)/((dfx1**2+dfy1**2+0.1)**(3/2)))
    for i in range(len(K)):
        if K[i] < 0.6:
            K[i] = 0

    # 可视化
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(contour[:, 0, 0], contour[:, 0, 1])
    plt.title('Original Contour')
    plt.subplot(1, 2, 2)
    plt.scatter(x, y, c=K, cmap='viridis')
    plt.colorbar(label='Pixel Values')
    plt.title('K Contour')
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
        ksize = 5
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
