from delta_Go.game_mechanics import GoEnv, choose_move_pass, transition_function


def test_ko():
    """this function tests that the ko rules in Go works correctly."""

    state, _, _, _ = GoEnv(choose_move_pass).reset()
    state = transition_function(state, action=0)
    assert state.ko is None
    state = transition_function(state, action=1)
    assert state.ko is None
    state = transition_function(state, action=2)
    assert state.ko is None
