import cv2
import numpy as np
from PIL import Image


def bound2line(classMap_numpy: np.ndarray, max_classes=-1, g_min_point=200, o_min_points=50,
               g_epsilon=0.003, o_epsilon=0.002):
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
        ret, thresh = cv2.threshold(mask, 1, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Areas smaller than the threshold are judged to be noise.
        if cls == 1:
            min_points = g_min_point
            epsilon = g_epsilon
        else:
            min_points = o_min_points
            epsilon = o_epsilon
        contours = [x for x in contours if len(x) >= min_points]
        if len(contours) > 0:
            approx_list[cls] = []
            for cnt in contours:
                arc_length = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon * arc_length, True).squeeze()
                approx_list[cls].append(approx)

    return approx_list
