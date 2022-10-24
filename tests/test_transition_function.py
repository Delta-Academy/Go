import cProfile
import random
import sys
from pathlib import Path

import numpy as np

import pytest
from delta_Go.game_mechanics import GoEnv, choose_move_randomly, transition_function
from tqdm import tqdm

HERE = Path(__file__).parent.parent.resolve()

sys.path.append(str(HERE))
sys.path.append(str(HERE / "delta_Go"))


def test_transition_function_takes_move():
    env = GoEnv(choose_move_randomly)
    env.reset()

    transition_env = transition_function(
        env,
        # Random move without passing
        action=choose_move_randomly(env.observation, env.legal_moves[:-1]),
    )

    assert not np.array_equal(env.observation, transition_env.observation)


def choose_move():
    pass


def test_transition_function_takes_correct_move():
    # TODO!!!!
    pass
    # env = GoEnv(choose_move_randomly)
