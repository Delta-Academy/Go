from delta_Go.game_mechanics import GoEnv, choose_move_pass, transition_function


def test_ko() -> None:
    """this function tests that the ko rules in Go works correctly."""
    state, _, _, _ = GoEnv(choose_move_pass).reset()
    # state = transition_function(state, action=3)
    state = transition_function(state, action=2)

    assert state.ko is None
    state = transition_function(state, action=5)

    assert state.ko is None
    state = transition_function(state, action=12)

    assert state.ko is None
    state = transition_function(state, action=13)

    assert state.ko is None
    state = transition_function(state, action=4)

    assert state.ko is None
    state = transition_function(state, action=3)

    assert state.ko is not None
