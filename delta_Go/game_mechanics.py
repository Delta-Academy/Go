import pickle
import random
import sys
import time
from copy import copy, deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

import pygame
import torch
from gym.spaces import Box, Discrete
from pettingzoo.classic.go.go_base import Position
from pettingzoo.classic.go_v5 import raw_env
from pettingzoo.utils import BaseWrapper
from pygame import Surface
from torch import nn

HERE = Path(__file__).parent.resolve()

# Hack as this won't pip install on replit
sys.path.append(str(HERE / "PettingZoo"))


BOARD_SIZE = 9
ALL_POSSIBLE_MOVES = np.arange(BOARD_SIZE**2 + 1)


# The komi to use is much debated. 7.5 seems to
# generalise well for different board sizes
# lifein19x19.com/viewtopic.php?f=15&t=17750
# 7.5 is also the komi used in alpha-go vs Lee Sedol
# (non-integer means there are no draws)

KOMI = 7.5


def play_go(
    your_choose_move: Callable,
    opponent_choose_move: Callable,
    game_speed_multiplier: float = 1.0,
    render: bool = True,
    verbose: bool = False,
) -> float:

    env = GoEnv(
        opponent_choose_move,
        verbose=verbose,
        render=render,
        game_speed_multiplier=game_speed_multiplier,
    )

    observation, reward, done, info = env.reset()
    done = False
    while not done:
        action = your_choose_move(observation=observation, legal_moves=info["legal_moves"], env=env)
        observation, reward, done, info = env.step(action)
        print(observation)
    return reward


class DeltaEnv(BaseWrapper):
    def __init__(
        self,
        env,
        opponent_choose_move: Callable,
        verbose: bool = False,
        render: bool = False,
        game_speed_multiplier: float = 1.0,
    ):

        super().__init__(env)

        self.opponent_choose_move = opponent_choose_move
        if render:
            pygame.init()
        self.render = render
        self.verbose = verbose
        self.game_speed_multiplier = game_speed_multiplier
        self.action_space = Discrete(BOARD_SIZE**2 + 1)
        self.observation_space = Box(low=-1, high=1, shape=(BOARD_SIZE, BOARD_SIZE))
        self.num_envs = 1

        # Which color do we play as
        self.player = random.choice(["black_0", "white_0"])
        self.first_game = True

    @property
    def turn(self) -> str:
        return self.env.agent_selection

    @property
    def turn_pretty(self) -> str:
        return self.turn[:-2]

    @property
    def observation(self):
        obs = self.env.last()[0]["observation"]
        player = obs[:, :, 0].astype("int")
        opponent = obs[:, :, 1].astype("int")
        return player - opponent

    @property
    def legal_moves(self):
        mask = self.env.last()[0]["action_mask"]
        full_space = self.env.action_space(self.turn)
        return np.arange(full_space.n)[mask.astype(bool)]

    @property
    def info(self) -> Dict:
        return {"legal_moves": self.legal_moves}

    @property
    def done(self) -> bool:
        return self.env.last()[2]

    def render_game(self, screen: Optional[Surface] = None) -> None:

        self.env.render(screen_override=screen)
        time.sleep(1 / self.game_speed_multiplier)

    def reset(self) -> Tuple[np.ndarray, float, bool, Dict]:
        # Only choose a new first player if not the first game
        if not self.first_game:
            self.player = random.choice(["black_0", "white_0"])
        super().reset()

        if self.verbose:
            print(
                f"Resetting Game.\nYou are playing with the {self.player[:-2]} tiles.\nBlack plays first\n\n"
            )

        if self.turn != self.player:
            self._step(
                self.opponent_choose_move(
                    observation=self.observation, legal_moves=self.legal_moves, env=self
                ),
            )

        return self.observation, 0, self.done, self.info

    def move_to_string(self, move: int) -> str:
        N = self.observation.shape[0]
        if move == N**2:
            return "passes"
        return f"places counter at coordinate: {(move//N, move%N)}"

    def __str__(self) -> str:
        return str(self.observation) + "\n"

    def _step(self, move: int) -> float:

        assert not self.done, "Game is done! Please reset() the env before calling step() again"
        assert move in self.legal_moves, f"{move} is an illegal move"

        if self.verbose:
            print(f"{self.turn_pretty} {self.move_to_string(move)}")

        self.env.step(move)

        if self.render:
            self.render_game()

        return self.reward

    @property
    def reward(self) -> float:
        return self.env.last()[1]

    @property
    def player_score(self) -> float:
        black_score = self.env.env.env.env.go_game.score()  # lol
        return black_score if self.player == "black_0" else black_score * -1

    def step(self, move: int) -> Tuple[np.ndarray, float, bool, Dict]:

        # Flipped because the env takes the step, changes the player, then we return the reward
        reward = -self._step(move)

        if not self.done:
            opponent_reward = self._step(
                self.opponent_choose_move(
                    observation=self.observation, legal_moves=self.legal_moves, env=self
                ),
            )
            # Flipped as above
            reward = opponent_reward

        if self.verbose and self.done:
            white_idx = int(self.turn_pretty == "white")
            black_idx = int(self.turn_pretty == "black")
            white_count = np.sum(self.env.last()[0]["observation"].astype("int")[:, :, white_idx])
            black_count = np.sum(self.env.last()[0]["observation"].astype("int")[:, :, black_idx])

            print(
                f"\nGame over. Reward = {reward}.\n"
                f"Player was playing as {self.player[:-2]}.\n"
                f"White has {white_count} counters.\n"
                f"Black has {black_count} counters.\n"
                f"Your score is {self.player_score}.\n"
            )
            self.first_game = False

        return self.observation, reward, self.done, self.info


def GoEnv(
    opponent_choose_move: Callable[[np.ndarray, np.ndarray], int],
    verbose: bool = False,
    render: bool = False,
    game_speed_multiplier: float = 1.0,
) -> DeltaEnv:
    return DeltaEnv(
        raw_env(board_size=BOARD_SIZE, komi=KOMI),
        opponent_choose_move,
        verbose,
        render,
        game_speed_multiplier=game_speed_multiplier,
    )


def choose_move_randomly(
    observation: np.ndarray, legal_moves: np.ndarray, env: Optional[DeltaEnv] = None
) -> int:
    return random.choice(legal_moves)


def choose_move_pass(observation: np.ndarray, legal_moves: np.ndarray, env: DeltaEnv) -> int:
    """passes on every turn."""
    return BOARD_SIZE**2


def int_to_coord(move: int):
    return (move // BOARD_SIZE, move % BOARD_SIZE)


def transition_function(state: Position, action) -> Position:
    """Transition function for the game of Go."""
    if action == BOARD_SIZE**2:
        action = None  # go_base pass representation
    else:
        action = int_to_coord(action)
    return state.play_move(action)


def reward_function(env: DeltaEnv) -> float:
    return env.reward


def load_pkl(team_name: str, network_folder: Path = HERE) -> nn.Module:
    net_path = network_folder / f"{team_name}_file.pkl"
    assert (
        net_path.exists()
    ), f"Network saved using TEAM_NAME='{team_name}' doesn't exist! ({net_path})"
    with open(net_path, "rb") as handle:
        file = pickle.load(handle)
    return file


def save_pkl(file: Any, team_name: str) -> None:
    assert "/" not in team_name, "Invalid TEAM_NAME. '/' are illegal in TEAM_NAME"
    net_path = HERE / f"{team_name}_file.pkl"
    n_retries = 5
    for attempt in range(n_retries):
        try:
            with open(net_path, "wb") as handle:
                pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
            load_pkl(team_name)
            return
        except Exception:
            if attempt == n_retries - 1:
                raise


# Need to know the default screen size from petting zoo to get which square is clicked
# Will not work with a screen override
PETTING_ZOO_SCREEN_SIZE = (500, 500)
SQUARE_SIZE = PETTING_ZOO_SCREEN_SIZE[0] // BOARD_SIZE
LEFT = 1


def pos_to_coord(pos: Tuple[int, int]) -> Tuple[int, int]:  # Assume square board
    col = pos[0] // SQUARE_SIZE
    row = pos[1] // SQUARE_SIZE
    return row, col


def coord_to_int(coord: Tuple[int, int]) -> int:
    return coord[0] * BOARD_SIZE + coord[1]


def human_player(state: np.ndarray, legal_moves: np.ndarray, env: DeltaEnv) -> int:

    print("Your move, click to place a tile!")

    while True:
        ev = pygame.event.get()
        for event in ev:
            if event.type == pygame.MOUSEBUTTONUP and event.button == LEFT:
                coord = pos_to_coord(pygame.mouse.get_pos())
                action = coord_to_int(coord)
                if action in legal_moves:
                    return action
