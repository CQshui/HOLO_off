import pywt
import cv2
import numpy as np
import os


# This function does the coefficient fusing according to the fusion method
def fuseCoeff(coeff1, coeff2, method):
    global coeff
    if (method == 'mean'):
        coeff = (coeff1 + coeff2) / 2
    elif (method == 'min'):
        coeff = np.minimum(coeff1, coeff2)
    elif (method == 'max'):
        coeff = np.maximum(coeff1, coeff2)
    return coeff


# 文件读取
image_path = "D:\\Desktop\\test\\dof"
file_names = os.listdir(image_path)
image_files = [f for f in file_names if any(ext in f.lower() for ext in ('.jpg', '.jpeg', '.png'))]
dof_img = cv2.imread(os.path.join(image_path, file_names[0]), 0)
# 打开并显示每张图片
for image_name in image_files:
    # 构建完整的文件路径
    image_path = "D:\\Desktop\\test\\dof"
    image_path = os.path.join(image_path, image_name)
    try:
        temp_img = cv2.imread(image_path, 0)
        # temp_img = cv2.resize(temp_img, (4508, 4096))
        # DOF操作
        # Params
        FUSION_METHOD = 'mean'  # Can be 'min' || 'max || anything you choose according theory
        FUSION_METHOD1 = 'max'
        FUSION_METHOD2 = 'min'
        # First: Do wavelet transform on each image
        wavelet = 'db2'
        coeff1 = pywt.wavedec2(dof_img[:, :], wavelet, level=1)
        coeff2 = pywt.wavedec2(temp_img[:, :], wavelet, level=1)
        # Second: for each level in both image do the fusion according to the desire option
        fusedCooef = []
        for i in range(len(coeff1)):
            # The first values in each decomposition is the apprximation values of the top level
            if (i == 0):
                fusedCooef.append(fuseCoeff(coeff1[0], coeff2[0], FUSION_METHOD))
            else:
                # For the rest of the levels we have tupels with 3 coefficients
                c1 = fuseCoeff(coeff1[i][0], coeff2[i][0], FUSION_METHOD1)
                c2 = fuseCoeff(coeff1[i][1], coeff2[i][1], FUSION_METHOD1)
                c3 = fuseCoeff(coeff1[i][2], coeff2[i][2], FUSION_METHOD1)
                fusedCooef.append((c1, c2, c3))
        # Third: After we fused the coefficient we need to transform back to get the image
        fusedImage = pywt.waverec2(fusedCooef, wavelet)
        # Forth: normalize values to be in uint8
        fusedImage1 = np.multiply(np.divide(fusedImage - np.min(fusedImage), (np.max(fusedImage) - np.min(fusedImage))),
                                  255)
        fusedImage1 = fusedImage1.astype(np.uint8)
        dof_img = fusedImage1

    except Exception as e:
        print("无法加载图片：", image_name)
        print("错误信息：", str(e))

cv2.imwrite("D:\\Desktop\\test\\dof\\DOF.bmp", dof_img)
