import numpy as np
import gym
from gym import spaces
from PIL import Image
from time import time

from einops import rearrange

import torchvision.transforms as T
from muse.core import constants, utils

# avoid warnings about box cast to float32
gym.logger.set_level(40)

# always use np_random and np.ramdom to ensure proper seeding


def random_xy(n, np_random, workspace, z_pos, min_dist=0.03):
    valid_samples = False
    while not valid_samples:
        pos = np_random.uniform(low=workspace[0], high=workspace[1], size=(n, 3))
        pos[:, 2] = z_pos
        # dist = np.linalg.norm(pos[:, None] - pos[None, :], axis=2)
        dist = np.abs(pos[:, None] - pos[None, :])[..., 0]
        if n == 1:
            valid_samples = True
        else:
            if min_dist:
                valid_samples = (dist[dist > 0]).min() > min_dist
            else:
                valid_samples = True

    quat = np.array([1.0, 0.0, 0.0, 0.0])
    quat = np.repeat(quat[None], n, axis=0)
    pos = np.hstack((pos, quat))
    return pos


def random_xytheta(
    n, np_random, workspace, z_pos, min_dist=0.03, theta_range=(-180, 180)
):
    pos = random_xy(n, np_random, workspace, z_pos, min_dist=min_dist)
    low, high = theta_range
    theta = np_random.uniform(low=low, high=high, size=(n, 1))
    euler = np.zeros((n, 3))

    euler[:, -1] = theta[:, 0]
    quat = np.zeros((n, 4))
    for i in range(n):
        quat[i] = utils.std_to_muj_quat(utils.euler_to_quat(euler[i], degrees=True))
    pos = np.hstack((pos[:, :3], quat))
    return pos


def create_action_space(scene, *keys):
    action_space = dict()
    for key in keys:
        dtype = np.float32
        if "linear_velocity" == key:
            high = np.full(3, scene.max_tool_velocity[0])
            low = -high
        elif "xy_linear_velocity" == key:
            high = np.full(2, scene.max_tool_velocity[0])
            low = -high
        elif "angular_velocity" == key:
            high = np.full(3, scene.max_tool_velocity[1])
            low = -high
        elif "theta_velocity" == key:
            high = np.array([scene.max_tool_velocity[1]])
            low = -high
        elif "grip_open" == key:
            high = np.full(1, constants.GRIP_OPEN)
            low = np.full(1, constants.GRIP_CLOSE)
        action_space[key] = spaces.Box(low=low, high=high, dtype=dtype)

    action_space = spaces.Dict(action_space)
    return action_space


def resize(im, size, im_type="rgb"):
    w, h = size
    pil_im = Image.fromarray(im)
    if im_type == "rgb":
        pil_im = pil_im.resize((w, h), Image.BILINEAR)
    elif im_type == "seg":
        pil_im = pil_im.resize((w, h), Image.NEAREST)
    else:
        raise ValueError(f"unknown im type: {im_type}")
    im = np.asarray(pil_im)
    return im


def realsense_resize_crop(im, im_type="rgb"):
    h, w = im.shape[:2]
    ori_w, ori_h = constants.REALSENSE_RESOLUTION
    # assert ori_h % h == 0 and ori_w % w == 0

    render_w, render_h = constants.RENDER_RESOLUTION
    if (w, h) != (render_w, render_h):
        im = resize(im, (render_w, render_h), im_type)

    h, w = im.shape[:2]
    bw, bh = constants.REALSENSE_CROP
    by = constants.REALSENSE_CROP_Y
    bx = (w - bw) // 2
    im = im[by : by + bh, bx : bx + bw]

    return im


def realsense_resize_batch_crop(im, im_type="rgb"):
    b, h, w = im.shape[:3]
    ori_w, ori_h = constants.REALSENSE_RESOLUTION
    # assert ori_h % h == 0 and ori_w % w == 0

    render_w, render_h = constants.RENDER_RESOLUTION

    if (w, h) != (render_w, render_h):
        start_t_resize = time()
        resized_im = T.Resize((render_h, render_w))(rearrange(im, "b h w c -> b c h w"))
        t_resize = time() - start_t_resize

    start_t_crop = time()
    h, w = resized_im.shape[-2:]
    bw, bh = constants.REALSENSE_CROP
    by = constants.REALSENSE_CROP_Y
    bx = (w - bw) // 2
    cropped_im = rearrange(
        resized_im[:, :, by : by + bh, bx : bx + bw], "b c h w -> b h w c"
    )
    t_crop = time() - start_t_crop

    # return im, [t_resize, t_crop]
    return cropped_im, [t_resize, t_crop]
