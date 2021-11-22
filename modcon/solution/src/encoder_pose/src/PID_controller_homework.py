#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import sys
sys.path.append('../')
from unit_test import UnitTestPositionPID

# Lateral control

# TODO: write the PID controller using what you've learned in the previous activities

# Note: y_hat will be calculated based on your DeltaPhi() and poseEstimate() functions written previously 

def PIDController(
    v_0, # assume given (by the scenario)
    y_ref, # assume given (by the scenario)
    y_hat, # assume given (by the odometry)
    prev_e_y, # assume given (by the previous iteration of this function)
    prev_int_y, # assume given (by the previous iteration of this function)
    delta_t): # assume given (by the simulator)
    """
    Args:
        v_0 (:double:) linear Duckiebot speed.
        y_ref (:double:) reference lateral pose
        y_hat (:double:) the current estiamted pose along y.
        prev_e_y (:double:) tracking error at previous iteration.
        prev_int_y (:double:) previous integral error term.
        delta_t (:double:) time interval since last call.
    returns:
        v_0 (:double:) linear velocity of the Duckiebot 
        omega (:double:) angular velocity of the Duckiebot
        e_y (:double:) current tracking error (automatically becomes prev_e_y at next iteration).
        e_int_y (:double:) current integral error (automatically becomes prev_int_y at next iteration).
    """
    
    # Tracking error
    e = y_ref - y_hat

    # integral of the error
    e_int_y = prev_int_y + e*delta_t

    # anti-windup - preventing the integral error from growing too much
    e_int_y = max(min(e_int_y,2),-2)

    # derivative of the error
    e_der = (e - prev_e_y)/delta_t

    # controller coefficients
    Kp = 0.1
    Kd = 5
    Ki = 0

    # PID controller for omega
    omega = Kp*e + Ki*e_int_y + Kd*e_der
    
    return [v_0, omega], e, e_int_y

def PIDController2(
    v_0, # assume given (by the scenario)
    y_ref, # assume given (by the scenario)
    y_hat, # assume given (by the odometry)
    prev_e_y, # assume given (by the previous iteration of this function)
    prev_int_y, # assume given (by the previous iteration of this function)
    delta_t, Kp, Ki, Kd): # assume given (by the simulator)
    
    # Tracking error
    e = y_ref - y_hat

    # integral of the error
    e_int_y = prev_int_y + e*delta_t

    # anti-windup - preventing the integral error from growing too much
    e_int_y = max(min(e_int_y,2),-2)

    # derivative of the error
    e_der = (e - prev_e_y)/delta_t

    # controller coefficients

    # PID controller for omega
    omega = Kp*e + Ki*e_int_y + Kd*e_der
    
    return [v_0, omega], e, e_int_y
R = 0.0318
baseline = 0.1
gain = 0.8
trim = 0
v_0 = 0.2
y_ref = 0.2


unit_test = UnitTestPositionPID(R, baseline, v_0, y_ref, gain, trim, PIDController) 
unit_test.test()

# unit test input/ R, baseline, v_0, gain, trim, PIDController
# unit_test = UnitTestPositionPID(R, baseline, v_0, y_ref, gain, trim, PIDController2)

# for kd in np.arange(4,5.5, 0.5):
#     # for kp in np.arange(0.1, 1, 0.05):
#     #     for ki in np.arange(0.1, 1, 0.05):
#     for kp in np.arange(4, 5.5, 0.5):
#     # kp = 0.1
#         for ki in np.arange(0,0.5, 0.1):
#                 # print('-----kp = {} ---- kd = {} --------'.format(kp, kd))
#             unit_test = UnitTestPositionPID(R, baseline, v_0, y_ref, gain, trim, PIDController2)
#             unit_test.test2(kp, ki, kd)
#         # print('-------------')

