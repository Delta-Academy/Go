from delta_Go.game_mechanics import BOARD_SIZE, is_terminal, transition_function
from delta_Go.state import State


def test_is_terminal() -> None:

    state = State()
    assert not is_terminal(state)
    state = transition_function(state, 0)
    assert not is_terminal(state)
    state = transition_function(state, 1)
    assert not is_terminal(state)
    state = transition_function(state, 10)
    assert not is_terminal(state)
    state = transition_function(state, 11)
    assert not is_terminal(state)
    state = transition_function(state, BOARD_SIZE**2)
    assert not is_terminal(state)
    state = transition_function(state, BOARD_SIZE**2)
    assert is_terminal(state)
