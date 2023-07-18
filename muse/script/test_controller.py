import time
import gym
import muse.envs

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

from muse.core.utils import euler_to_quat, std_to_muj_quat

SIM_DT = 0.1


def main():
    pick_env = gym.make("Pick-v0", cam_list=[], viewer=True)
    gripper_pos = [-0.40, 0, 0.1]
    real_obs = pick_env.reset(mocap_target_pos=gripper_pos)

    # pick_env.unwrapped.render()

    # print(std_to_muj_quat(euler_to_quat([np.pi, 0, 0], False)))

    error_nb = 0
    time_prev = time.time()
    time_list = []

    step_size = 0.1
    n_step = 5 / step_size

    theta_obs = []
    for i in range(int(n_step)):
        real_obs = pick_env.step(
            {
                "linear_velocity": np.array([0.0, 0, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, -3.14 / 10]),
                "grip_open": 1,
            }
        )
        theta_obs.append(real_obs[0]["gripper_theta"])

    plt.plot(theta_obs)

    theta_real_robot = [
        1.5710113221187632,
        1.5671979367920115,
        1.5195948296040196,
        1.4875485932966808,
        1.462979192144453,
        1.437529205104666,
        1.393343113773517,
        1.3734921706788485,
        1.32961060992573,
        1.2979374709191325,
        1.2799038224830774,
        1.2290161124144314,
        1.2100358342617707,
        1.1711345760133742,
        1.1319826206426655,
        1.1080083338668534,
        1.0785375500875256,
        1.0464146293547933,
        1.006806569672174,
        0.9824961262279354,
        0.9545983019841358,
        0.9205069878102765,
        0.8824411480989434,
        0.8589557911125222,
        0.8206177923309914,
        0.7956229940725196,
        0.765906728951222,
        0.7324471662860519,
        0.7136456351357772,
        0.662107389749526,
        0.6316617583771457,
        0.6061489730688687,
        0.581120300333152,
        0.5462688743581389,
        0.5284057718175597,
        0.4771841994252936,
        0.44618446885283136,
        0.41325146872201785,
        0.38210573836845835,
        0.35803396641578655,
        0.33193674273429047,
        0.29436521997434767,
        0.27007551040929195,
        0.2310830840166995,
        0.19988410431157969,
        0.1678823961954442,
        0.12974145118313601,
        0.10493432323382759,
        0.07963047275917287,
        0.03583364950945987,
    ]

    plt.plot(theta_real_robot)
    plt.show()
    print(f"Gripper should have moved {n_step*step_size*SIM_DT} m in Y")

    time.sleep(10000)


if __name__ == "__main__":
    main()
