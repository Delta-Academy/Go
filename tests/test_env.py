import random

import numpy as np

from delta_go.game_mechanics import (
    BLACK,
    BOARD_SIZE,
    KOMI,
    GoEnv,
    choose_move_pass,
    choose_move_randomly,
    play_go,
)
from delta_go.go_base import all_legal_moves, score
from delta_go.state import State
from delta_go.utils import MAX_NUM_MOVES

PASS_MOVE = BOARD_SIZE**2


def choose_move_pass_at_end(state):
    legal_moves = all_legal_moves(state.board, state.ko)
    if len(legal_moves) < 10:
        return PASS_MOVE
    non_pass_moves = legal_moves[:-1]
    return non_pass_moves[int(random.random()) * len(non_pass_moves)]


def test_play_go():
    for _ in range(10):

        reward = play_go(
            your_choose_move=choose_move_pass_at_end,
            opponent_choose_move=choose_move_pass,
            game_speed_multiplier=100,
            render=False,
            verbose=False,
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
    legal_moves = all_legal_moves(state.board, state.ko)
    assert max(legal_moves) <= BOARD_SIZE**2
    assert min(legal_moves) >= 0


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


def choose_move_top_left(state: State) -> int:
    legal_moves = all_legal_moves(state.board, state.ko)
    return 0 if 0 in legal_moves else choose_move_randomly(state)


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
    legal_moves = all_legal_moves(state.board, state.ko)
    assert max(legal_moves) <= BOARD_SIZE**2
    assert min(legal_moves) >= 0
    assert len(legal_moves) <= BOARD_SIZE**2 - 1

    board = state.board * env.player_color

    assert board[0, 0] == -1
    assert board[0, 1] == 1

    if reset_took_step:
        # This fails if the reset passed but happens rarely
        assert np.sum(board != 0) == 3
    else:
        assert np.sum(board != 0) == 2
        assert np.all(board.ravel()[2:] == 0)


def test_env_game_over() -> None:
    env = GoEnv(
        choose_move_randomly,
    )
    state, reward, done, info = env.reset()
    while not done:
        action = choose_move_randomly(state=state)
        state, reward, done, info = env.step(action)

    assert done
    assert action == PASS_MOVE or len(state.recent_moves) == MAX_NUM_MOVES

    # Probably can be refactored but oh well
    score_ = score(state.board, KOMI)
    if score_ > 0 and state.player_color == BLACK:
        assert reward == 1
    elif score_ < 0 and state.player_color == BLACK:
        assert reward == -1
    elif score_ > 0:
        assert reward == -1
    elif score_ < 0:
        assert reward == 1


def test_env_game_over_n_times():
    for _ in range(10):
        test_env_game_over()
