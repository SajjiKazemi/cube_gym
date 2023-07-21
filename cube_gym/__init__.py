from gym.envs.registration import register

register(
    id='cube_gym/CubeGym-v0',
    entry_point='cube_gym.envs:CubeGym',
)