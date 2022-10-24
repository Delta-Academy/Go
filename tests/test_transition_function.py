import cProfile
import random
import sys
from pathlib import Path

import numpy as np
import pytest
from tqdm import tqdm

from delta_Go.game_mechanics import (
    BOARD_SIZE,
    GoEnv,
    choose_move_pass,
    choose_move_randomly,
    transition_function,
)
from pettingzoo.classic.go.go_base import Position

HERE = Path(__file__).parent.parent.resolve()
sys.path.append(str(HERE))
sys.path.append(str(HERE / "delta_Go"))


def test_transition_function_takes_move():
    env = GoEnv(choose_move_pass)
    env.reset()

    new_state = transition_function(env.state, action=0)
    assert not np.array_equal(env.state.board, new_state.board)


def choose_move_not_top_left(legal_moves, **kwargs):
    legal_moves = legal_moves[legal_moves != 0]

    if len(legal_moves) > 1:  # Don't pass if you don't have to
        legal_moves = legal_moves[legal_moves != BOARD_SIZE**2]

    return random.choice(legal_moves)


def test_transition_function_correct_move():

    env = GoEnv(choose_move_pass)  # Will leave a new game board after reset
    env.reset()

    starting_state = env.state
    starting_board = starting_state.board
    assert np.all(starting_board == 0)

    # Make the same move on the transition function and the env
    # Should give the same result
    new_state = transition_function(starting_state, action=0)
    new_board = new_state.board
    env.step(move=0)

    # Check the state hasn't changed in place
    assert np.all(starting_state.board == 0)

    assert new_board[0, 0] == env.player_color

    np.testing.assert_array_equal(env.state.board, new_board)


# if __name__ == "main":
#     test_transition_function_correct_move()
