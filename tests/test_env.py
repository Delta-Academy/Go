import random

import numpy as np

from delta_Go.game_mechanics import (
    BOARD_SIZE,
    GoEnv,
    choose_move_pass,
    choose_move_randomly,
    play_go,
    transition_function,
)

PASS_MOVE = BOARD_SIZE**2


def choose_move_pass_at_end(legal_moves, **kwargs):
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
            verbose=False,
        )

        # Playing against a passer
        assert reward == 1


def test_env_reset():

    env = GoEnv(
        choose_move_randomly,
    )

    observation, reward, done, info = env.reset()
    assert done == False
    assert reward == 0
    assert observation.shape == (BOARD_SIZE, BOARD_SIZE)
    assert max(info["legal_moves"]) <= BOARD_SIZE**2
    assert min(info["legal_moves"]) >= 0


def test_env__step():
    env = GoEnv(
        choose_move_pass,
    )
    observation, reward, done, info = env.reset()

    assert env.turn == env.player

    assert np.all(observation == 0)
    reward = env._step(move=0)
    assert reward == 0

    new_observation = env.observation
    # 1s from the player's perspective
    assert new_observation[0, 0] == 1
    new_observation[0, 0] = 0
    assert np.all(new_observation == 0)
    assert env.reward == 0
    assert env.done == False


def choose_move_top_left(legal_moves, **kwargs):
    return 0 if 0 in legal_moves else random.choice(legal_moves)


def test_env_step():

    env = GoEnv(
        choose_move_top_left,
    )
    observation, reward, done, info = env.reset()

    assert observation[0, 0] in [0, -1]

    reset_took_step = observation[0, 0] != 0

    if not reset_took_step:
        assert np.all(observation == 0)

    assert env.turn == env.player

    observation, reward, done, info = env.step(1)
    assert done == False
    assert reward == 0
    assert max(info["legal_moves"]) <= BOARD_SIZE**2
    assert min(info["legal_moves"]) >= 0
    assert len(info["legal_moves"]) <= BOARD_SIZE**2 - 1

    assert observation[0, 0] == -1
    assert observation[0, 1] == 1

    if reset_took_step:
        assert np.sum(observation != 0) == 3
    else:
        assert np.sum(observation != 0) == 2
        assert np.all(observation.ravel()[2:] == 0)


def test_env_game_over():

    env = GoEnv(
        choose_move_randomly,
    )
    observation, reward, done, info = env.reset()
    while not done:
        action = choose_move_randomly(observation=observation, legal_moves=info["legal_moves"])
        observation, reward, done, info = env.step(action)

    assert done
    assert action == PASS_MOVE
    if np.sum(observation == 1) > np.sum(observation == -1):
        assert reward == 1
    elif np.sum(observation == 1) < np.sum(observation == -1):
        assert reward == -1
    else:
        assert reward == 0  # Draw
