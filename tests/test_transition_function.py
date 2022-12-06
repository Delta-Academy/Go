import random

import numpy as np

from delta_go.game_mechanics import (
    PASS_MOVE,
    GoEnv,
    choose_move_pass,
    choose_move_randomly,
    is_terminal,
    reward_function,
    transition_function,
)
from delta_go.go_base import all_legal_moves, game_over
from delta_go.state import State
from delta_go.utils import BOARD_SIZE, MAX_NUM_MOVES


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
    # Got to be careful of suicide moves! These will leave the board the same!
    state, _, _, _ = GoEnv(choose_move_pass).reset()  # Will leave a new game board after reset

    # Place in every location from top left to bottom right, row by row (except last row)
    for action in range(BOARD_SIZE * (BOARD_SIZE - 1)):
        new_state = transition_function(state, action=action)
        assert not np.array_equal(new_state.board, state.board)
        state = new_state


def test_transition_function_no_in_place_mutation_10_times() -> None:
    for _ in range(10):
        test_transition_function_no_in_place_mutation()


def test_ensure_to_play_changes_terminal():
    state = State()
    state = transition_function(state, choose_move_randomly(state))  # Black
    state = transition_function(state, 81)  # White
    state = transition_function(state, 81)  # Black
    # Black wins
    assert is_terminal(state)
    assert reward_function(state) == 1  # Since player_move is BLACK by default


def test_max_num_moves_is_terminal():
    state = State()
    for _ in range(MAX_NUM_MOVES // 2):
        # Black plays randomly non-pass moves
        legal_moves = all_legal_moves(state.board, state.ko)
        state = transition_function(
            state, legal_moves[int(random.random() * (len(legal_moves) - 1))]
        )  # black

        # white - passes unless board about to be full no legal moves
        legal_moves = all_legal_moves(state.board, state.ko)
        state = transition_function(state, 81 if len(legal_moves) > 2 else legal_moves[0])

    assert is_terminal(state)
    assert reward_function(state) != 0
