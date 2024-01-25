from skimage.filters import threshold_local
from skimage.restoration import inpaint
import numpy as np
import scipy.ndimage as sn
# 思路是在颗粒部位蒙上蒙版，然后用inpaint_biharmonic函数补全，得到用于补偿的相位图
# 非常失败，内存不够，不知道行不行得通


def phase_compensate(U):
    local_thresh = threshold_local(U, block_size=5, offset=1)
    binary_U = U > local_thresh
    binary_U = np.int64(binary_U)

    kernel2 = np.ones((10, 10), np.uint8)
    U_ero2 = sn.binary_erosion(binary_U, structure=kernel2, iterations=5)
    U_dil2 = sn.binary_dilation(U_ero2, structure=kernel2, iterations=5)

    U_dil1 = sn.binary_dilation(U_dil2, iterations=5)
    U_ero1 = sn.binary_erosion(U_dil1, iterations=5)

    mask = U_ero1
    u = inpaint.inpaint_biharmonic(U, mask=mask)
    return u
