import random
import sys
from pathlib import Path

import numpy as np

from delta_Go.game_mechanics import (
    BLACK,
    BOARD_SIZE,
    GoEnv,
    Position,
    choose_move_pass,
    choose_move_randomly,
    play_go,
    transition_function,
)

HERE = Path(__file__).parent.parent.resolve()
sys.path.append(str(HERE))
sys.path.append(str(HERE / "delta_Go"))


PASS_MOVE = BOARD_SIZE**2


def choose_move_pass_at_end(state):
    legal_moves = state.legal_moves
    if len(legal_moves) < 10:
        return PASS_MOVE
    return random.choice(legal_moves[legal_moves != BOARD_SIZE**2])


def test_play_go():
    for _ in range(10):

        reward = play_go(
            your_choose_move=choose_move_pass_at_end,
            opponent_choose_move=choose_move_pass,
            game_speed_multiplier=100,
            render=False,
            verbose=True,
        )

        assert reward == 1


def test_env_reset():

    env = GoEnv(
        choose_move_randomly,
    )

    state, reward, done, info = env.reset()
    assert done == False
    assert reward == 0
    assert state.board.shape == (BOARD_SIZE, BOARD_SIZE)
    assert max(state.legal_moves) <= BOARD_SIZE**2
    assert min(state.legal_moves) >= 0


def test_env__step() -> None:
    env = GoEnv(
        choose_move_pass,
    )
    state, reward, done, info = env.reset()

    assert state.to_play == env.player_color

    assert np.all(state.board == 0)
    env._step(move=0)

    new_board = env.state.board * env.player_color

    # 1s from the player's perspective
    assert new_board[0, 0] == 1
    new_board[0, 0] = 0
    assert np.all(new_board == 0)
    assert env.reward == 0
    assert env.done == False


def choose_move_top_left(state: Position):
    legal_moves = state.legal_moves
    return 0 if 0 in legal_moves else random.choice(legal_moves)


def test_env_step() -> None:

    env = GoEnv(
        choose_move_top_left,
    )
    state, reward, done, info = env.reset()

    board = state.board * env.player_color

    assert board[0, 0] in [0, -1]

    reset_took_step = board[0, 0] != 0

    if not reset_took_step:
        assert np.all(board == 0)

    assert env.state.to_play == env.player_color

    state, reward, done, info = env.step(1)
    assert done == False
    assert reward == 0
    assert max(state.legal_moves) <= BOARD_SIZE**2
    assert min(state.legal_moves) >= 0
    assert len(state.legal_moves) <= BOARD_SIZE**2 - 1

    board = state.board * env.player_color

    assert board[0, 0] == -1
    assert board[0, 1] == 1

    if reset_took_step:
        assert np.sum(board != 0) == 3
    else:
        assert np.sum(board != 0) == 2
        assert np.all(board.ravel()[2:] == 0)


def test_env_game_over():

    env = GoEnv(
        choose_move_randomly,
    )
    state, reward, done, info = env.reset()
    while not done:
        action = choose_move_randomly(state=state)
        state, reward, done, info = env.step(action)

    board = state.board * env.player_color
    assert done
    assert action == PASS_MOVE

    # Probably can be refactored but oh well
    if state.score() > 0 and state.player_color == BLACK:
        assert reward == 1
    elif state.score() < 0 and state.player_color == BLACK:
        assert reward == -1
    elif state.score() > 0 and state.player_color != BLACK:
        assert reward == -1
    elif state.score() < 0 and state.player_color != BLACK:
        assert reward == 1


def test_env_game_over_n_times():
    for _ in range(10):
        test_env_game_over()
