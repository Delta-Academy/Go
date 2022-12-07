from delta_go.game_mechanics import BOARD_SIZE, State, is_terminal, transition_function


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
