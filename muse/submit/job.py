import os
from pathlib import Path
import subprocess
import argparse
import numpy as np
from colorama import Fore, Style
from sklearn.model_selection import ParameterGrid

from muse.submit.template import sched_template, collect_template
from muse.submit.config import (
    sched_cfg,
    base_cfg,
    dev_cfg,
    task_cfgs,
)


def recursive_format(cfg):
    flat_cfg = cfg.copy()
    for k, v in flat_cfg.items():
        if isinstance(v, str):
            while "{" in v:
                if k in v:
                    raise ValueError(f"{k}: auto-referencement in the cfg file.")
                v = v.format(**flat_cfg)
            flat_cfg[k] = v
    return flat_cfg


def load_config(partial_cfg):
    full_cfg = sched_cfg.copy()

    full_cfg.update(base_cfg)
    if partial_cfg.pop("dev"):
        full_cfg.update(dev_cfg)

    task = partial_cfg["task"]
    task_cfg = task_cfgs[task]
    full_cfg.update(task_cfg)

    full_cfg.update(partial_cfg)

    return full_cfg


def print_config(cfg, keys):
    if not keys:
        keys = sorted(cfg.keys())
    for k in keys:
        v = cfg[k]
        print(f"{Fore.YELLOW}{k}{Style.RESET_ALL}: {v}")


def launch_job(template, cfg, keys, sub):
    job_cfg = recursive_format(cfg)
    template = template.format(**job_cfg)

    template_path = "submit.slurm"
    with open(template_path, "w") as f:
        f.write(template)
    os.chmod(template_path, 0o755)

    print_config(job_cfg, keys)
    if sub:
        args_run = ["sbatch", template_path]
        subprocess.run(args_run)
        print(f"{job_cfg['job_name']} submitted.")
    else:
        print(template)
    os.remove(template_path)
    return job_cfg


def grid_submit(template, exp_cfgs, keys, sub):
    for k, v in exp_cfgs.items():
        if not isinstance(v, list):
            exp_cfgs[k] = [v]

    cfg_grid = ParameterGrid(exp_cfgs)
    cfg_grid = list(cfg_grid)
    print("GRID")
    print(cfg_grid)
    if len(cfg_grid) == 1:
        keys = []

    for i, partial_cfg in enumerate(cfg_grid):
        print(f"{i}:")
        full_cfg = load_config(partial_cfg)
        job_cfg = launch_job(template, full_cfg, keys, sub)


def parse_user_args(user_args):
    exp_cfgs = dict()
    assert len(user_args) % 2 == 0
    for i in range(0, len(user_args), 2):
        k = user_args[i]
        assert k[:2] == "--"
        k = k[2:]
        k = k.replace("-", "_")
        v = user_args[i + 1]
        if "," in v:
            v = v.split(",")
        exp_cfgs[k] = v
    return exp_cfgs


def get_args_parser():
    parser = argparse.ArgumentParser("Job launch script", add_help=False)
    parser.add_argument("--job-name", type=str)
    parser.add_argument("--nodes", type=int, default=0)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--dev", dest="dev", action="store_true")
    # whether to submit the job or not
    parser.add_argument("--no-sub", dest="sub", action="store_false")
    parser.set_defaults(dev=False, sub=True)
    return parser


def main(args, exp_cfgs):
    if args.suffix:
        args.suffix = "_" + args.suffix
    exp_name = "{job_name}" + args.suffix

    template = sched_template + "\n"
    template += collect_template

    exp_cfgs["job_name"] = args.job_name
    exp_cfgs["exp_name"] = exp_name
    exp_cfgs["dev"] = args.dev

    keys = []
    # grid_submit(template, exp_cfgs, keys, args.sub)
    # grid_submit(
    #     template,
    #     exp_cfgs,
    #     dict(light_rand_factor=np.arange(0, 10, 0.5).tolist()),
    #     args.sub,
    # )
    # grid_submit(
    #     template,
    #     exp_cfgs,
    #     dict(hsv_rand_factor=np.arange(0, 10, 0.5).tolist()),
    #     args.sub,
    # )
    # grid_submit(
    #     template,
    #     exp_cfgs,
    #     dict(cam_rand_factor=np.arange(0, 10, 0.5).tolist()),
    #     args.sub,
    # )
    # grid_submit(
    #     template,
    #     exp_cfgs,
    #     dict(num_textures=[100, 250, 500, 1000, 2500, 5000, 10000, 12500, 15000]),
    #     args.sub,
    # )
    grid_submit(
        template,
        exp_cfgs,
        keys,
        args.sub,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Job launch script", parents=[get_args_parser()])
    args, user_args = parser.parse_known_args()
    exp_cfgs = parse_user_args(user_args)
    main(args, exp_cfgs)
