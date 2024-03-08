import cv2


def hull_solid(contour):
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    area = cv2.contourArea(contour)

    judge = cv2.isContourConvex(contour)

    return judge


def defect(contour):
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)
    num_large_defects = 0

    depth_threshold = 100*cv2.contourArea(contour)/cv2.arcLength(contour, True)
    if defects is not None:
        large_defects = [d for d in defects if d[0][3] > depth_threshold]  # 过滤小的凸起
        num_large_defects = len(large_defects)  # 大凸起瑕疵的个数
    print(defects)

    return num_large_defects
