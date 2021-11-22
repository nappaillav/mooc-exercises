from typing import Tuple

import numpy as np


def get_motor_left_matrix(shape):
    res = np.zeros(shape=shape, dtype="float32")  # write your function instead of this one
    W_S, W, W_E = int(shape[1]*0.1), int(shape[1]*0.5), int(shape[1]*0.9)
    H = int(0.6*shape[0])
    res[H:, W_S:W] = np.linspace(0,1, shape[0]-H).repeat(W-W_S).reshape(shape[0]-H,W-W_S)
    # res[H:, W:W_E] = -np.linspace(0,1, shape[0]-H).repeat(W_E-W).reshape(shape[0]-H,W_E-W)
    # res[300:, 200:] = 1
    return res

def get_motor_right_matrix(shape):
    res = np.zeros(shape=shape, dtype="float32")  # write your function instead of this one
    W_S, W, W_E = int(shape[1]*0.1), int(shape[1]*0.5), int(shape[1]*0.9)
    H = int(shape[0]*0.6)
    # res[H:, W_S:W] = -np.linspace(0,1, shape[0]-H).repeat(W-W_S).reshape(shape[0]-H,W-W_S)
    res[H:, W:W_E] = np.linspace(0,1,shape[0]-H).repeat(W_E-W).reshape(shape[0]-H,W_E-W)
    return res
