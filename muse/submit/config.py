"""
Scheduler cfg
"""

sched_cfg = dict(
    qos="qos_gpu-t3",
    time="03:00:00",
    nodes=1,
    gpus_per_node=1,
    cpus_per_gpu=80,
    conda_env_name="muse",
    conda_dir="$WORK/miniconda3/",
    queue="",
    # queue="#SBATCH -C v100-32g",
)

dev_cfg = dict(qos="qos_gpu-dev", time="02:00:00")

"""
Base cfg
"""

base_cfg = dict(
    output_dir="{output_dir}",
    seed=0,
    env_name="DR-Pick-v0",
    num_workers=50,
    light_rand_factor=2.0,
    light_pos_rand_factor=5.0,
    xyz_cam_rand_factor=4.0,
    rpy_cam_rand_factor=2.0,
    fov_cam_rand_factor=1.0,
    hsv_rand_factor=3.0,
    num_textures=15000,
    seed_textures=10,
    seed_camera=0,
    noise_over_calibration_factor=0.0,
    num_distractors=0,
)


"""
Task cfgs
"""

task_cfgs = dict(
    poses=dict(
        keyword_samples="poses",
        samples=20000,
    ),
    demos=dict(
        keyword_samples="episodes",
        samples=100,
    ),
)
