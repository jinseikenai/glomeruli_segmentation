import cv2
import numpy as np
from PIL import Image


def bound2line(classMap_numpy: np.ndarray, max_classes=-1, g_min_area=10000, o_min_area=2500,
               g_epsilon=0.003, o_epsilon=0.008):
    """
    Extract the boundaries of each class from a classMap: ndarray

    :param classMap_numpy:
    :param max_classes: If you want to ignore the mesangium area, max_class is set to 4.
    :param g_min_point:
    :param o_min_points:
    :param g_epsilon:
    :param o_epsilon:
    :return:
    """
    if max_classes < 0:
        num_class = classMap_numpy.max()+1
    else:
        num_class = min(max_classes, classMap_numpy.max()+1)
    # ignore background area. The value 0 is set in the background.
    approx_list = {}
    for cls in range(1, num_class):
        if cls == 1:
            mask = (classMap_numpy >= cls).astype(np.uint8) * 255
        else:
            mask = (classMap_numpy == cls).astype(np.uint8) * 255
        # img = Image.fromarray(mask)
        # img.show()
        # 二値画像を作って、その輪郭を抽出する(findContours)
        ret, thresh = cv2.threshold(mask, 1, 255, 0)
        # 輪郭を抽出する前にカスレ対応で膨張(Dilate)させる
        thresh = cv2.dilate(thresh, kernel=np.ones((3, 3), np.uint8), iterations=2)
        # 輪郭を抽出する前にカスレ対応で縮小(erosion)させる
        # thresh = cv2.erode(thresh, kernel=np.ones((3, 3), np.uint8), iterations=2)
        # cv2.RETR_EXTERNAL を指定して、入れ子の Contour は無視する
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Areas smaller than the threshold are judged to be noise.
        if cls == 1:
            min_area = g_min_area
        else:
            min_area = o_min_area
        contours = [x for x in contours if cv2.contourArea(x) >= min_area]
        if len(contours) > 0:
            approx_list[cls] = []
            for cnt in contours:
                arc_length = cv2.arcLength(cnt, True)
                area = cv2.contourArea(cnt)
                if area > g_min_area:
                    epsilon = g_epsilon
                else:
                    epsilon = o_epsilon
                approx = cv2.approxPolyDP(cnt, epsilon * arc_length, True).squeeze()
                '''
                if cls == 1:
                    arc_length = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon * arc_length, True).squeeze()
                else:
                    approx = cv2.approxPolyDP(cnt, epsilon * area, True).squeeze()
                '''
                approx_list[cls].append(approx)

    return approx_list
