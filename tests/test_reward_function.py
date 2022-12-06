from delta_go.game_mechanics import reward_function, transition_function
from delta_go.go_base import BOARD_SIZE
from delta_go.state import State
from delta_go.utils import BLACK, WHITE


def test_reward_function_black_loses() -> None:

    state = State()
    assert reward_function(state) == 0
    state = transition_function(state, 0)
    assert reward_function(state) == 0
    state = transition_function(state, 1)
    assert reward_function(state) == 0
    state = transition_function(state, 10)
    assert reward_function(state) == 0
    state = transition_function(state, 11)
    assert reward_function(state) == 0
    state = transition_function(state, BOARD_SIZE**2)
    assert reward_function(state) == 0
    state = transition_function(state, BOARD_SIZE**2)
    # Two stones each, komi makes white win
    assert reward_function(state) == -1


def test_reward_function_black_wins() -> None:

    white_move = BOARD_SIZE**2

    state = State()
    assert reward_function(state) == 0

    for black_move in range(81):

        assert state.to_play == BLACK
        state = transition_function(state, black_move)
        assert reward_function(state) == 0

        assert state.to_play == WHITE
        state = transition_function(state, white_move)
        assert reward_function(state) == 0

    assert state.to_play == BLACK
    state = transition_function(state, BOARD_SIZE**2)
    assert reward_function(state) == 1
