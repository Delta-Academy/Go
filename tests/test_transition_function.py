import random

import numpy as np
from tqdm import tqdm

import pytest
from delta_Go.game_mechanics import GoEnv, choose_move_randomly, transition_function


def test_transition_function():
    env = GoEnv(choose_move_randomly)
    env.reset()
    transition_env = transition_function(
        env, action=choose_move_randomly(env.observation, env.legal_moves)
    )
    assert not np.array_equal(env.observation, transition_env.observation)


def test_test_transition_function():
    for _ in tqdm(range(1000)):
        test_transition_function()
