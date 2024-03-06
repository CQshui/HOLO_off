import cv2


def hull_solid(contour):
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    area = cv2.contourArea(contour)

    return float(area)/hull_area


def defect(contour):
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)

    return defects.shape[0]/cv2.arcLength(contour, True)*100
