from typing import Callable
from env_wrapper import DeltaEnv

import numpy as np

from pettingzoo.classic import go_v5

BOARD_SIZE = 3

ALL_POSSIBLE_MOVES = np.arange(BOARD_SIZE**2 + 1)


def GoEnv(
    opponent_choose_move: Callable[[np.ndarray, np.ndarray], int],
    verbose: bool = False,
    render: bool = False,
    game_speed_multiplier: int = 0,
):
    return DeltaEnv(
        go_v5.env(board_size=BOARD_SIZE, komi=0),
        opponent_choose_move,
        verbose,
        render,
        game_speed_multiplier=game_speed_multiplier,
    )


def choose_move_randomly(observation, legal_moves):
    return np.random.choice(legal_moves)


def choose_move_pass(observation, legal_moves) -> int:
    "passes on every turn"
    return BOARD_SIZE**2


def play_go(
    your_choose_move: Callable,
    opponent_choose_move: Callable,
    game_speed_multiplier=1,
    render=True,
    verbose=False,
) -> None:

    env = GoEnv(
        opponent_choose_move,
        verbose=verbose,
        render=render,
        game_speed_multiplier=game_speed_multiplier,
    )

    observation, reward, done, info = env.reset()
    while not done:
        action = your_choose_move(observation, info["legal_moves"])
        observation, reward, done, info = env.step(action)
