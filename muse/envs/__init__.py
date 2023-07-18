from gym.envs.registration import register

envs = [
    dict(
        id="Pick-v0",
        entry_point="muse.envs.pick:PickEnv",
        max_episode_steps=150,
        reward_threshold=1.0,
    ),
    dict(
        id="Stack-v0",
        entry_point="muse.envs.stack:StackEnv",
        max_episode_steps=800,
        reward_threshold=1.0,
    ),
    dict(
        id="StackButtons-v0",
        entry_point="muse.envs.stack_buttons:StackButtonsEnv",
        max_episode_steps=800,
        reward_threshold=1.0,
    ),
    dict(
        id="MultimodalStack-v0",
        entry_point="muse.envs.multimodal_stack:MultimodalStackEnv",
        max_episode_steps=800,
        reward_threshold=1.0,
    ),
    dict(
        id="Push-v0",
        entry_point="muse.envs.push:PushEnv",
        max_episode_steps=600,
        reward_threshold=1.0,
    ),
    dict(
        id="MultimodalPush-v0",
        entry_point="muse.envs.multimodal_push:MultimodalPushEnv",
        max_episode_steps=600,
        reward_threshold=1.0,
    ),
    dict(
        id="Reach-v0",
        entry_point="muse.envs.reach:ReachEnv",
        max_episode_steps=100,
        reward_threshold=1.0,
    ),
    dict(
        id="PushAndPick-v0",
        entry_point="muse.envs.push_and_pick:PushAndPickEnv",
        max_episode_steps=600,
        reward_threshold=1.0,
    ),
    dict(
        id="Bowl-v0",
        entry_point="muse.envs.bowl:BowlEnv",
        max_episode_steps=600,
        reward_threshold=1.0,
    ),
    dict(
        id="RopeShaping-v0",
        entry_point="muse.envs.rope_shaping:RopeShapingEnv",
        max_episode_steps=1000,
        reward_threshold=1.0,
    ),
    dict(
        id="Sweep-v0",
        entry_point="muse.envs.sweep:SweepEnv",
        max_episode_steps=2000,
        reward_threshold=1.0,
    ),
    dict(
        id="Assembly-v0",
        entry_point="muse.envs.assembly:AssemblyEnv",
        max_episode_steps=600,
        reward_threshold=1.0,
    ),
    dict(
        id="BoxRetrieving-v0",
        entry_point="muse.envs.box_retrieving:BoxRetrievingEnv",
        max_episode_steps=800,
        reward_threshold=1.0,
    ),
    dict(
        id="BoxButtons-v0",
        entry_point="muse.envs.box_buttons:BoxButtonsEnv",
        max_episode_steps=800,
        reward_threshold=1.0,
    ),
    dict(
        id="PushButtons-v0",
        entry_point="muse.envs.push_buttons:PushButtonsEnv",
        max_episode_steps=2000,
        reward_threshold=1.0,
    ),
]


num_variants = 5220

for env_dict in envs:
    name = env_dict["id"]
    if name == "PushButtons-v0":
        num_variants = 5220
        for variant_id in range(num_variants):
            var_env_dict = env_dict.copy()
            var_name = f"Var{variant_id}-{name}"
            var_env_dict["id"] = var_name
            var_env_dict["kwargs"] = dict(variant_id=variant_id)
            register(**var_env_dict)
            # domain randomization
            dr_name = f"DR-{var_name}"
            dr_env_dict = var_env_dict.copy()
            dr_env_dict["id"] = dr_name
            dr_env_dict["kwargs"] = dict(
                domain_randomization=True, variant_id=variant_id
            )
            register(**dr_env_dict)
    else:
        register(**env_dict)
        # domain randomization
        dr_name = f"DR-{name}"
        env_dict = env_dict.copy()
        env_dict["id"] = dr_name
        env_dict["kwargs"] = dict(domain_randomization=True)
        register(**env_dict)
