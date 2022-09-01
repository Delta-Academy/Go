import random
import time
from typing import Callable, Dict, Optional

import numpy as np
from gym.spaces import Box, Discrete

from pettingzoo.utils import BaseWrapper


class DeltaEnv(BaseWrapper):
    def __init__(
        self,
        env,
        opponent_choose_move: Callable,
        verbose: bool = False,
        render: bool = False,
        game_speed_multiplier: int = 1,
    ):

        super().__init__(env)

        self.opponent_choose_move = opponent_choose_move
        self.render = render
        self.verbose = verbose
        self.game_speed_multiplier = game_speed_multiplier

    @property
    def turn(self) -> str:
        return self.env.agent_selection

    @property
    def turn_pretty(self) -> str:
        return self.turn[:-2]

    @property
    def observation(self):
        # For some reason the depth of the third dimesion is 17? Keep an eye
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

    def render_game(self) -> None:

        self.env.render()
        time.sleep(1 / self.game_speed_multiplier)

    def reset(self):
        super().reset()

        # Which color do we play as
        self.player = random.choice(["black_0", "white_0"])
        if self.verbose:
            print(
                f"Resetting Game.\nYou are playing with the {self.player[:-2]} tiles.\nBlack plays first\n\n"
            )

        if self.turn != self.player:
            self._step(
                self.opponent_choose_move(
                    observation=self.observation, legal_moves=self.legal_moves
                ),
            )

        return self.observation, 0, self.done, self.info

    def move_to_string(self, move: int):
        N = self.observation.shape[0]
        if move == N**2:
            return "passes"
        return f"places counter at coordinate: {(move//N, move%N)}"

    def __str__(self):
        return str(self.observation) + "\n"

    def _step(self, move: int) -> float:

        assert not self.done, "Game is done! Please reset() the env before calling step() again"
        assert move in self.legal_moves, f"{move} is an illegal move"

        if self.verbose:
            print(f"{self.turn_pretty} {self.move_to_string(move)}")

        self.env.step(move)
        print(self)

        if self.render:
            self.render_game()

        return self.reward

    @property
    def reward(self):
        return self.env.last()[1]

    def step(self, move: int):

        reward = -self._step(move)

        if not self.done:
            opponent_reward = self._step(
                self.opponent_choose_move(
                    observation=self.observation, legal_moves=self.legal_moves
                ),
            )
            reward = opponent_reward

        if self.done:
            if self.verbose:
                white_idx = int(self.turn_pretty == "white")
                black_idx = int(self.turn_pretty == "black")

                black_score = self.env.env.env.env.go_game.score()  # lol
                result_string = self.env.env.env.env.go_game.result_string()

                player_score = black_score if self.player == "black_0" else black_score * -1

                white_count = np.sum(
                    self.env.last()[0]["observation"].astype("int")[:, :, white_idx]
                )
                black_count = np.sum(
                    self.env.last()[0]["observation"].astype("int")[:, :, black_idx]
                )
                print(
                    f"Game over. Reward = {reward}. White has {white_count} counters. "
                    f"Player was playing as {self.player}. "
                    f"Black has {black_count} counters. "
                    f"Your score was {player_score}. "
                    f"Black score was {black_score}. "
                    f"Result string was {result_string}"
                )
                time.sleep(1000)

        return self.observation, reward, self.done, self.info
