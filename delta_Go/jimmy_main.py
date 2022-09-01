import copy
import time
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from check_submission import check_submission
from game_mechanics import BOARD_SIZE, GoEnv, choose_move_pass, choose_move_randomly, play_go
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean

TEAM_NAME = "Team Jimmy"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"


HERE = Path(__file__).parent.resolve()


class ChooseMoveCheckpoint:
    def __init__(self, checkpoint_name: str):
        self.neural_network = copy.deepcopy(MaskablePPO.load(HERE / checkpoint_name))

    def choose_move(self, observation, legal_moves):
        neural_network = self.neural_network
        mask = np.isin(np.arange(BOARD_SIZE**2 + 1), legal_moves)
        action, _ = neural_network.predict(observation, deterministic=False, action_masks=mask)
        return action


class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.rewards = []

    def _on_step(self) -> None:
        """Called every step()"""
        self.rewards.append(safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]))


# def mask_fn(env):
#     return env.last()[0]["action_mask"]


# def smooth_trace(trace: np.ndarray, one_sided_window_size: int = 3) -> np.ndarray:
#     """Smooths a trace by averaging over a window of size one_sided_window_size."""
#     window_size = int(2 * one_sided_window_size + 1)
#     trace[one_sided_window_size:-one_sided_window_size] = (
#         np.convolve(trace, np.ones(window_size), mode="valid") / window_size
#     )
#     return trace


def mask_fn(env):
    mask = np.repeat(False, BOARD_SIZE**2 + 1)
    mask[env.legal_moves] = True
    return mask


def smooth_trace(trace: np.ndarray, one_sided_window_size: int = 3) -> np.ndarray:
    """Smooths a trace by averaging over a window of size one_sided_window_size."""
    window_size = int(2 * one_sided_window_size + 1)
    trace[one_sided_window_size:-one_sided_window_size] = (
        np.convolve(trace, np.ones(window_size), mode="valid") / window_size
    )
    return trace


def train() -> Dict:

    for idx in range(16, 22):
        checkpoint = ChooseMoveCheckpoint(f"checkpoint{idx}")
        env = GoEnv(checkpoint.choose_move, verbose=False, render=False)
        env.reset_only_observation = True
        env = ActionMasker(env, mask_fn)
        env.reset()

        # model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=2, ent_coef=0.01)
        model = MaskablePPO.load(HERE / f"checkpoint{idx}")
        model.set_env(env)

        callback = CustomCallback()
        model.learn(total_timesteps=150_000, callback=callback)

        model.save(HERE / f"checkpoint{idx+1}")

    plt.plot(smooth_trace(callback.rewards, 100))
    plt.draw()
    plt.axhline(0)
    plt.show()
    return model


#
#     # test_model(model)

#     ################
#     # Play checkpointed self

#     choose_move_checkpoint = ChooseMoveCheckpoint("round6").choose_move

#     env = PokerEnv(choose_move_checkpoint, verbose=True, render=False)
#     env = ActionMasker(env, mask_fn)
#     env.reset()

#     # model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=2, ent_coef=0.01)
#     model = MaskablePPO.load("round6")
#     model.set_env(env)

#     callback = CustomCallback()
#     model.learn(total_timesteps=500_000, callback=callback)

#     model.save("round7")
#     print("reached checkpoint \n\n\n\n\n")
#     plt.plot(smooth_trace(callback.rewards, 2048))
#     plt.title(f"mean: {np.nanmean(callback.rewards)}, std: {np.nanstd(callback.rewards)}")
#     plt.axhline(0)

#     plt.show()
#     return model


def n_games(player, opponent):

    rewards = []
    for _ in tqdm(range(100)):
        reward = play_go(
            your_choose_move=player,
            opponent_choose_move=opponent,
            game_speed_multiplier=100,
            render=False,
            verbose=False,
        )
        rewards.append(reward)
        if reward == 1:
            print("win")
        else:
            print("loss")
    print(np.mean(rewards))


def choose_move(state: np.ndarray, legal_moves: np.ndarray, neural_network: nn.Module) -> int:
    """Called during competitive play. It acts greedily given current state of the board and your
    network. It returns a single move to play.

    Args:
         state: The state of poker game. shape = (72,)
         legal_moves: Legal actions on this turn. Subset of {0, 1, 2, 3}
         neural_network: Your pytorch network from train()

    Returns:
        action: Single value drawn from legal_moves
    """

    neural_network = MaskablePPO.load(HERE / "checkpoint3")
    mask = np.isin(np.arange(BOARD_SIZE**2 + 1), legal_moves)
    action, _ = neural_network.predict(state, deterministic=False, action_masks=mask)
    return action


if __name__ == "__main__":

    ## Example workflow, feel free to edit this! ###
    # file = train()
    # save_pkl(file, TEAM_NAME)

    # check_submission(
    #     TEAM_NAME
    # )  # <---- Make sure I pass! Or your solution will not work in the tournament!!

    # my_network = load_pkl(TEAM_NAME)

    # # Code below plays a single game against a random
    # #  opponent, think about how you might want to adapt this to
    # #  test the performance of your algorithm.
    def choose_move_no_network(state: Any, legal_moves) -> int:
        """The arguments in play_game() require functions that only take the state as input.

        This converts choose_move() to that format.
        """
        return choose_move(state, legal_moves, None)

    # opponent = ChooseMoveCheckpoint("checkpoint20").choose_move
    player = ChooseMoveCheckpoint("checkpoint21").choose_move
    opponent = choose_move_randomly

    play_go(
        your_choose_move=player,
        opponent_choose_move=opponent,
        game_speed_multiplier=10,
        render=True,
        verbose=True,
    )

    train()
    # n_games(player=player, opponent=opponent)
