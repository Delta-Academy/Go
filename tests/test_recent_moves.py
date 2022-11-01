import numpy as np

from delta_go.game_mechanics import BOARD_SIZE, GoEnv, choose_move_pass, transition_function
from delta_go.utils import BLACK, WHITE


def test_recent_moves():
    env = GoEnv(choose_move_pass)  # Will leave a new game board after reset
    env.reset()

    starting_state = env.state

    assert len(starting_state.recent_moves) in {0, 1}
    if len(starting_state.recent_moves) == 1:
        assert starting_state.recent_moves[0].move == BOARD_SIZE**2
        assert starting_state.recent_moves[0].color == BLACK
        reset_takes_move = True
    else:
        reset_takes_move = False

    new_state = transition_function(starting_state, action=0)

    if reset_takes_move:
        assert len(starting_state.recent_moves) == 1
        assert len(new_state.recent_moves) == 2
        assert new_state.recent_moves[0].move == BOARD_SIZE**2
        assert new_state.recent_moves[0].color == BLACK
        assert new_state.recent_moves[1].move == 0
        assert new_state.recent_moves[1].color == WHITE
    else:
        assert len(starting_state.recent_moves) == 0
        assert len(new_state.recent_moves) == 1
        assert new_state.recent_moves[0].move == 0
        assert new_state.recent_moves[0].color == BLACK


def test_test_recent_moves_10_times():
    for _ in range(10):
        test_recent_moves()
