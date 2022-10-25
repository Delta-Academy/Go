import random
from pathlib import Path

import numpy as np

from delta_Go.game_mechanics import (
    BOARD_SIZE,
    PASS_MOVE,
    GoEnv,
    choose_move_pass,
    transition_function,
)
from delta_Go.go_base import all_legal_moves, game_over


def test_transition_function_takes_move():
    env = GoEnv(choose_move_pass)
    env.reset()

    new_state = transition_function(env.state, action=0)
    assert not np.array_equal(env.state.board, new_state.board)


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


def test_transition_function_no_in_place_mutation() -> None:

    state, _, _, _ = GoEnv(choose_move_pass).reset()  # Will leave a new game board after reset
    while not game_over(state.recent_moves):
        action = random.choice(all_legal_moves(state.board, state.ko))

        new_state = transition_function(state, action=action)
        if action != PASS_MOVE:
            assert not np.array_equal(new_state.board, state.board)
        state = new_state


def test_transition_function_no_in_place_mutation_10_times() -> None:
    for _ in range(10):
        test_transition_function_no_in_place_mutation()
