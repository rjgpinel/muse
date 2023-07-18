# simulation
SIM_STEPS = 10
RENDER_STEPS = 4
# WARMUP_STEPS = 100
WARMUP_STEPS = 1000
# control the number of steps to close the gripper
# GRIPPER_ACTION_SCALING = 0.04
# TODO: tune
GRIPPER_ACTION_SCALING = 0.05

# controller rate - second
CONTROLLER_DT = 0.1
# (linvel, angvel) - meter / second
MAX_TOOL_VELOCITY = (0.07, 0.25)
MAX_TOOL_PUSH_VELOCITY = (0.035, 0.25)

# gripper action values
GRIP_OPEN = 1.0
GRIP_CLOSE = -1.0
# gripper opening/closing length in script - second
GRIP_ACTION_LENGTH = 4

# camera
REALSENSE_RESOLUTION = (1280, 720)
# same 16/9 aspect ratio than original resolution
RENDER_RESOLUTION = (448, 252)
# RENDER_RESOLUTION = (856, 504) # For larger rendering
# 4/3 crop of the image
REALSENSE_CROP = (240, 180)
# REALSENSE_CROP = (480, 360) # For larger rendering

REALSENSE_CROP_Y = 8
REALSENSE_FOV = 42.5

# Textures
TEXTURES_PATH = dict(
    bop="textures/bop/", dtd="textures/dtd/", kth="textures/kth-kyberge-uiuc"
)
