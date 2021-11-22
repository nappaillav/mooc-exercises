import numpy as np
import cv2

lower_hsv = np.array([171, 140, 100])
upper_hsv = np.array([179, 200, 255])
yellow_lower = np.array([10, 100, 0])
yellow_upper = np.array([70, 255, 255])

def preprocess(image_rgb: np.ndarray) -> np.ndarray:
    """ Returns a 2D array """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    mask_1 = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask_2 = cv2.inRange(hsv, yellow_lower, yellow_upper)
    mask = np.bitwise_or(mask_1, mask_2)
    #     masked = cv2.bitwise_or(image, image, mask=mask)
    return mask
