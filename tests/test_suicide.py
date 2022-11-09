from delta_go.game_mechanics import transition_function
from delta_go.state import State
from delta_go.utils import BLACK, EMPTY, WHITE


def test_top_suicide():
    for player_suicide in [BLACK, WHITE]:
        state = State()

        if player_suicide == BLACK:
            state = transition_function(state, 81)  # Pass

        # Setup shown below: X's and O's are stones, S is suicide position
        # O S O
        #   O
        state = transition_function(state, 0)  # Top left
        state = transition_function(state, 81)  # Pass
        state = transition_function(state, 2)  # Top, 2 in from left
        state = transition_function(state, 81)  # Pass
        state = transition_function(state, 10)  # 2nd row, 1 in from left

        # Pre-suicide, the 1st space should be empty
        assert state.board[0, 1] == EMPTY
        # Take suicide move
        state = transition_function(state, 1)

        # Post-suicide, the 1st space should be empty
        assert state.board[0, 1] == EMPTY


def test_middle_suicide():
    for player_suicide in [BLACK, WHITE]:
        state = State()

        if player_suicide == BLACK:
            state = transition_function(state, 81)  # Pass

        # Setup shown below: X's and O's are stones, S is suicide position
        #   O
        # O S O
        #   O
        state = transition_function(state, 1)
        state = transition_function(state, 81)  # Pass
        state = transition_function(state, 9)
        state = transition_function(state, 81)  # Pass
        state = transition_function(state, 11)
        state = transition_function(state, 81)  # Pass
        state = transition_function(state, 19)

        # Pre-suicide, the hole should be empty
        assert state.board[1, 1] == EMPTY
        # Take suicide move
        state = transition_function(state, 10)

        # Post-suicide, the eye should be empty
        assert state.board[1, 1] == EMPTY


def test_two_space_suicide():
    for player_suicide in [BLACK, WHITE]:
        state = State()

        if player_suicide == BLACK:
            state = transition_function(state, 81)  # Pass

        # Setup shown below: X's and O's are stones, S is suicide position
        #   O O
        # O X S O
        #   O O
        state = transition_function(state, 1)
        state = transition_function(state, 81)  # Pass
        state = transition_function(state, 2)
        state = transition_function(state, 81)  # Pass
        state = transition_function(state, 9)
        state = transition_function(state, 81)  # Pass
        state = transition_function(state, 12)
        state = transition_function(state, 10)
        state = transition_function(state, 19)
        state = transition_function(state, 81)  # Pass
        state = transition_function(state, 20)

        # Pre-suicide, the hole should be empty
        assert state.board[1, 1] == player_suicide
        assert state.board[1, 2] == EMPTY
        # Take suicide move
        state = transition_function(state, 11)

        # Post-suicide, the eye should be empty and other piece removed
        assert state.board[1, 1] == EMPTY
        assert state.board[1, 2] == EMPTY


def test_three_space_suicide():
    for player_suicide in [BLACK, WHITE]:
        state = State()

        if player_suicide == BLACK:
            state = transition_function(state, 81)  # Pass

        # Setup shown below: X's and O's are stones, S is suicide position
        #   O O O
        # O X S X O
        #   O O O
        state = transition_function(state, 1)
        state = transition_function(state, 81)  # Pass
        state = transition_function(state, 2)
        state = transition_function(state, 81)  # Pass
        state = transition_function(state, 3)
        state = transition_function(state, 81)  # Pass
        state = transition_function(state, 9)
        state = transition_function(state, 10)
        state = transition_function(state, 13)
        state = transition_function(state, 12)
        state = transition_function(state, 19)
        state = transition_function(state, 81)  # Pass
        state = transition_function(state, 20)
        state = transition_function(state, 81)  # Pass
        state = transition_function(state, 21)

        assert state.board[1, 1] == player_suicide
        assert state.board[1, 2] == EMPTY
        assert state.board[1, 3] == player_suicide

        # Play suicide
        state = transition_function(state, 11)  # White

        assert state.board[1, 1] == EMPTY
        assert state.board[1, 2] == EMPTY
        assert state.board[1, 3] == EMPTY


def test_corner_suicide():
    for player_suicide in [BLACK, WHITE]:
        state = State()

        if player_suicide == BLACK:
            state = transition_function(state, 81)  # Pass

        # Setup shown below: X's and O's are stones, S is suicide position
        # _______
        # | S X
        # | X
        state = transition_function(state, 1)
        state = transition_function(state, 81)  # Pass
        state = transition_function(state, 9)

        assert state.board[0, 0] == EMPTY

        state = transition_function(state, 0)

        assert state.board[0, 0] == EMPTY


def test_diagonal_suicide():
    for player_suicide in [BLACK, WHITE]:
        state = State()

        if player_suicide == BLACK:
            state = transition_function(state, 81)  # Pass

        # Setup shown below: X's and O's are stones, S is suicide position
        # __________
        # |   X
        # | X   X
        # |   X S X
        # |     X O
        state = transition_function(state, 1)
        state = transition_function(state, 81)  # Pass
        state = transition_function(state, 9)
        state = transition_function(state, 81)  # Pass

        state = transition_function(state, 11)
        state = transition_function(state, 81)  # Pass
        state = transition_function(state, 19)
        state = transition_function(state, 81)  # Pass

        state = transition_function(state, 21)
        state = transition_function(state, 30)
        state = transition_function(state, 29)

        # Pre-suicide
        assert state.board[2, 2] == EMPTY
        state = transition_function(state, 20)

        # Post-suicide
        assert state.board[2, 2] == EMPTY


def test_l_shaped_suicide():
    for player_suicide in [BLACK, WHITE]:
        state = State()

        if player_suicide == BLACK:
            state = transition_function(state, 81)  # Pass

        # Setup shown below: X's and O's are stones, S is suicide position
        # __________
        # |   X X
        # | X O S X
        # |   X O X
        # |     X
        state = transition_function(state, 1)
        state = transition_function(state, 81)
        state = transition_function(state, 2)
        state = transition_function(state, 81)

        state = transition_function(state, 9)
        state = transition_function(state, 10)
        state = transition_function(state, 12)
        state = transition_function(state, 81)

        state = transition_function(state, 19)
        state = transition_function(state, 20)
        state = transition_function(state, 21)
        state = transition_function(state, 81)

        state = transition_function(state, 29)

        # Pre-suicide
        assert state.board[1, 1] == player_suicide
        assert state.board[1, 2] == EMPTY
        assert state.board[2, 2] == player_suicide
        state = transition_function(state, 11)

        # Post-suicide
        assert state.board[1, 1] == EMPTY
        assert state.board[1, 2] == EMPTY
        assert state.board[2, 2] == EMPTY
