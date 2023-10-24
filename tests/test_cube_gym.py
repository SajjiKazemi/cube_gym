import pytest
import gym

import sys
sys.path.append("../src")
from src.cube_gym.envs.cube_gym import *


class TestCubeGym:

    def test_CubeGym_init(self):
        env = CubeGym()
        assert env.size == 5