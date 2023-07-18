import numpy as np


def rotate(theta, block_to_gripper_dir_xy):
    rot_2d = theta_to_rotation2d(theta)
    return rot_2d @ block_to_gripper_dir_xy


def theta_to_rotation2d(theta):
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return r


def get_theta_from_vector(vector):
    return np.arctan2(vector[1], vector[0])
