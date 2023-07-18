sched_template = """#!/bin/bash
#SBATCH --job-name={job_name}

#SBATCH --qos={qos}
{queue}

#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:{gpus_per_node}
#SBATCH --cpus-per-task={cpus_per_gpu}

#SBATCH --time={time}

# cleaning modules launched during interactive mode
module purge

# conda
. {conda_dir}/etc/profile.d/conda.sh
export LD_LIBRARY_PATH={conda_dir}/envs/bin/lib:$LD_LIBRARY_PATH
export WANDB_API_KEY=d1e8f69de29481f3793656cb29f35e9c2b53e812
export WANDB_MODE="offline"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfslocalsup/spack_soft/mesa/18.3.6/gcc-8.3.1-prm3s2n7ixxt4vbajjp4z5ewfrwtuyya/lib
export LIBRARY_PATH=$LIBRARY_PATH:/gpfslocalsup/spack_soft/mesa/18.3.6/gcc-8.3.1-prm3s2n7ixxt4vbajjp4z5ewfrwtuyya/lib
export CPATH=$CPATH:/gpfslocalsup/spack_soft/mesa/18.3.6/gcc-8.3.1-prm3s2n7ixxt4vbajjp4z5ewfrwtuyya/include
module load cuda/11.2
cd $WORK/Code/mujoco-py/
python setup.py build
python setup.py install
"""
collect_template = """conda activate {conda_env_name}
mkdir -p {output_dir}/{seed_textures}/lf{light_rand_factor}-xyzf{xyz_cam_rand_factor}-rpyf{rpy_cam_rand_factor}-fovf{fov_cam_rand_factor}-hf{hsv_rand_factor}-nt{num_textures}/
srun --output {output_dir}/{seed_textures}/lf{light_rand_factor}-xyzf{xyz_cam_rand_factor}-rpyf{rpy_cam_rand_factor}-fovf{fov_cam_rand_factor}-hf{hsv_rand_factor}-nt{num_textures}/%j.out --error {output_dir}/{seed_textures}/lf{light_rand_factor}-xyzf{xyz_cam_rand_factor}-rpyf{rpy_cam_rand_factor}-fovf{fov_cam_rand_factor}-hf{hsv_rand_factor}-nt{num_textures}/%j.err \\
sh -c "
python -m ibc.collect.{task} \
  --output-dir {output_dir}/{seed_textures}/lf{light_rand_factor}-xyzf{xyz_cam_rand_factor}-rpyf{rpy_cam_rand_factor}-fovf{fov_cam_rand_factor}-hf{hsv_rand_factor}-nt{num_textures}/ \
  --{keyword_samples} {samples} \
  --seed {seed} \
  --light-rand-factor {light_rand_factor} \
  --xyz-cam-rand-factor {xyz_cam_rand_factor} \
  --rpy-cam-rand-factor {rpy_cam_rand_factor} \
  --fov-cam-rand-factor {fov_cam_rand_factor} \
  --hsv-rand-factor {hsv_rand_factor} \
  --num-workers {num_workers} \
  --env-name {env_name} \
  --num-textures {num_textures} \
  --seed-textures {seed_textures} \
  --seed-camera {seed_camera} \
  --noise-over-calibration-factor {noise_over_calibration_factor} \
  --num-distractors {num_distractors} \
"
"""
