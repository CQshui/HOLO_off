import pywt
import cv2
import numpy as np
import os


# This function does the coefficient fusing according to the fusion method
def fuseCoeff(coeff1, coeff2, method):
    global coeff
    if method == 'mean':
        coeff = (coeff1 + coeff2) / 2
    elif method == 'min':
        coeff = np.minimum(coeff1, coeff2)
    elif method == 'max':
        coeff = np.maximum(coeff1, coeff2)
    return coeff


def counting(count, total):
    count += 1
    print(count, '/', total)
    return count


def sobel(array):
    g_xh = cv2.Sobel(array[1][0], cv2.CV_64F, 1, 0)+cv2.Sobel(array[1][2], cv2.CV_64F, 1, 0)
    g_yh = cv2.Sobel(array[1][1], cv2.CV_64F, 0, 1)+cv2.Sobel(array[1][2], cv2.CV_64F, 0, 1)
    g_xl = cv2.Sobel(array[0], cv2.CV_64F, 1, 0)
    g_yl = cv2.Sobel(array[0], cv2.CV_64F, 0, 1)

    # g_h = cv2.addWeighted(g_xh, 0.5, g_yh, 0.5, 0)
    g_h = np.sqrt(g_xh**2+g_yh**2)
    g_l = np.sqrt(g_xl**2+g_yl**2)
    var_h = local_variance(g_h, 4005)
    var_l = local_variance(g_l, 4005)

    return var_h, var_l


def local_variance(array, block_size=15):
    # # 计算块的大小
    # block_size = max(block_size, 3)  # 确保块大小至少为3
    # block_size = block_size - (block_size % 2) + 1  # 确保块大小为奇数
    #
    # # 计算局部均值
    # mean_filter = np.ones((block_size, block_size)) / (block_size ** 2)
    # local_mean = cv2.filter2D(array, -1, mean_filter)
    #
    # # 计算局部方差
    # squared_image = np.square(array.astype(float))
    # squared_filter = mean_filter ** 2
    # # local_variance = np.sqrt(np.maximum(cv2.filter2D(squared_image, -1, squared_filter) - np.square(local_mean), 0))
    # local_variance = cv2.filter2D(squared_image, -1, squared_filter) - np.square(local_mean)

    kernel = np.ones((block_size, block_size))
    d1 = cv2.blur(array, (block_size, block_size))
    result1 = d1**2
    result2 = cv2.blur(array**2, (block_size, block_size))

    result0 = np.sqrt(np.maximum(result2 - result1, 0))

    return result0


def dof(input_pth, output_pth):
    # 文件读取
    image_path = input_pth
    file_names = os.listdir(image_path)
    image_files = [f for f in file_names if any(ext in f.lower() for ext in ('.jpg', '.jpeg', '.png', '.bmp'))]
    dof_img = cv2.imread(os.path.join(image_path, file_names[0]), 0)
    count = 0
    # 打开并显示每张图片
    for image_name in image_files:
        # 构建完整的文件路径
        image_path = input_pth
        image_path = os.path.join(image_path, image_name)
        temp_img = cv2.imread(image_path, 0)
        # First: Do wavelet transform on each image
        wavelet = 'db2'
        coeff1 = pywt.wavedec2(dof_img[:, :], wavelet, level=1)
        coeff2 = pywt.wavedec2(temp_img[:, :], wavelet, level=1)
        # temp_img = cv2.resize(temp_img, (4508, 4096))
        # DOF操作
        # Params
        FUSION_METHOD = 'mean'
        FUSION_METHOD1 = 'max'
        FUSION_METHOD2 = 'min'
        # Second: for each level in both image do the fusion according to the desire option
        fusedCooef = []
        Sobel_method = False
        if Sobel_method:
            # 得到亮度梯度的局部方差，1和2表示不同图片，h和l代表高频和低频
            Sobel_h1 = sobel(coeff1)[0]
            Sobel_l1 = sobel(coeff1)[1]
            Sobel_h2 = sobel(coeff2)[0]
            Sobel_l2 = sobel(coeff2)[1]

            # 判别矩阵，存放0或1
            judge_h1 = Sobel_h1.copy()
            judge_l1 = Sobel_l1.copy()

            for row in range(Sobel_h1.shape[0]):
                for col in range(Sobel_h1.shape[1]):
                    if Sobel_h1[row][col] >= Sobel_h2[row][col]:
                        judge_h1[row][col] = 1
                    else:
                        judge_h1[row][col] = 0

                    if Sobel_l1[row][col] >= Sobel_l2[row][col]:
                        judge_l1[row][col] = 1
                    else:
                        judge_l1[row][col] = 0

            judge_h2 = np.ones_like(judge_h1)-judge_h1
            judge_l2 = np.ones_like(judge_l1)-judge_l1

            # 四个小波系数
            c0 = coeff1[0]*judge_l1+coeff2[0]*judge_l2
            c1 = coeff1[1][0]*judge_h1+coeff2[1][0]*judge_h2
            c2 = coeff1[1][1]*judge_h1+coeff2[1][1]*judge_h2
            c3 = coeff1[1][2]*judge_h1+coeff2[1][2]*judge_h2
            fusedCooef.append(c0)
            fusedCooef.append((c1, c2, c3))

            # 小波重建
            fusedImage = pywt.waverec2(fusedCooef, wavelet)

            # 转为uint8格式
            fusedImage1 = np.multiply(np.divide(fusedImage - np.min(fusedImage), (np.max(fusedImage) - np.min(fusedImage))),
                                      255)
            fusedImage1 = fusedImage1.astype(np.uint8)
            dof_img = fusedImage1
            count = counting(count, len(image_files))

        else:
            for i in range(len(coeff1)):
                # The first values in each decomposition is the apprximation values of the top level
                if i == 0:
                    c0 = fuseCoeff(coeff1[0], coeff2[0], FUSION_METHOD)
                    fusedCooef.append(c0)
                else:
                    # For the rest of the levels we have tupels with 3 coefficients
                    c1 = fuseCoeff(coeff1[i][0], coeff2[i][0], FUSION_METHOD1)
                    c2 = fuseCoeff(coeff1[i][1], coeff2[i][1], FUSION_METHOD1)
                    c3 = fuseCoeff(coeff1[i][2], coeff2[i][2], FUSION_METHOD1)
                    fusedCooef.append((c1, c2, c3))
            # Third: After we fused the coefficient we need to transform back to get the image
            fusedImage = pywt.waverec2(fusedCooef, wavelet)
            # Forth: normalize values to be in uint8
            fusedImage1 = np.multiply(np.divide(fusedImage-np.min(fusedImage), (np.max(fusedImage)-np.min(fusedImage))), 255)
            fusedImage1 = fusedImage1.astype(np.uint8)
            dof_img = fusedImage1
            count = counting(count, len(image_files))
        # except Exception as e:
        #     print("无法加载图片：", image_name)
        #     print("错误信息：", str(e))

    cv2.imwrite(output_pth, dof_img)


if __name__ == "__main__":
    batch = True
    if not batch:
        input_path = "C:\\Users\\d1009\\Desktop\\test\\dof"
        output_path = "C:\\Users\\d1009\\Desktop\\test\\dof\\DOF.bmp"
        dof(input_path, output_path)

    else:
        directory = "F:/Data/20240329/Reconstruction/Limestone"
        img_num = 0
        img_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        for img_dir in img_dirs:
            img_num = img_num + 1
            img_directory = os.path.join(directory, img_dir)
            input_path = img_directory
            output_path = 'F:/Data/20240329/DOF/Limestone/Limestone_DOF_{}.bmp'.format(img_num)
            dof(input_path, output_path)


