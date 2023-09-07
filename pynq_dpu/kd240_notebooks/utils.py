from enum import Enum
from typing import List

import numpy as np
import cv2

def softmax(data) -> np.ndarray:
    """

    Args:
        data: ndarray on which we perform the softmax

    Returns: ndarray

    """
    data_sum: np.ndarray = np.zeros(data.shape)
    data_exp: np.ndarray = np.exp(data)
    data_sum[:, 0] = np.sum(data_exp, axis=1)
    data_sum[:, 1] = data_sum[:, 0]
    result: np.ndarray = data_exp / data_sum
    return result


def nms_boxes(boxes, scores, nms_threshold) -> list:
    """
    Args:
        nms_threshold: nms threshold
        boxes (np.ndarray): all the bounding boxes found
        scores (np.ndarray): all the scores for each bounding box

    Returns:
        list of "true" faces
    """
    x1: np.ndarray = boxes[:, 0]
    y1: np.ndarray = boxes[:, 1]
    x2: np.ndarray = boxes[:, 2]
    y2: np.ndarray = boxes[:, 3]
    areas: np.ndarray = (x2 - x1 + 1) * (y2 - y1 + 1)
    order: np.ndarray = scores.argsort()[::-1]
    keep: list = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w1 = np.maximum(0.0, xx2 - xx1 + 1)
        h1 = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w1 * h1
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_threshold)[0]  # threshold
        order = order[inds + 1]
    return keep

