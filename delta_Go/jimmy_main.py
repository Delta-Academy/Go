import time
from typing import Any, Dict
import matplotlib.pyplot as plt

from tqdm import tqdm

from check_submission import check_submission
from game_mechanics import GoEnv, choose_move_randomly, play_go, choose_move_pass

TEAM_NAME = "Team Jimmy"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"


# class ChooseMoveCheckpoint:
#     def __init__(self, checkpoint_name: str):
#         self.neural_network = copy.deepcopy(load_checkpoint(checkpoint_name))

#     def choose_move(self, state, legal_moves):
#         neural_network = self.neural_network
#         mask = np.isin(np.arange(4), legal_moves)
#         action, _ = neural_network.predict(state, deterministic=False, action_masks=mask)
#         return action


# class CustomCallback(BaseCallback):
#     def __init__(self, verbose=0):
#         super(CustomCallback, self).__init__(verbose)
#         self.rewards = []

#     def _on_step(self) -> None:
#         """Called every step()"""
#         self.rewards.append(safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]))


# def mask_fn(env):
#     return env.last()[0]["action_mask"]


# def smooth_trace(trace: np.ndarray, one_sided_window_size: int = 3) -> np.ndarray:
#     """Smooths a trace by averaging over a window of size one_sided_window_size."""
#     window_size = int(2 * one_sided_window_size + 1)
#     trace[one_sided_window_size:-one_sided_window_size] = (
#         np.convolve(trace, np.ones(window_size), mode="valid") / window_size
#     )
#     return trace


# def train() -> Dict:

#     #############
#     # Play against hard-coded opponent

#     # env = PokerEnv(choose_move_rules, verbose=True, render=False)
#     # env = ActionMasker(env, mask_fn)
#     # env.reset()

#     # model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=2, ent_coef=0.01)

#     # env.reset()

#     # # model = PPO.load("checkpoint1")
#     # # model.set_env(env)

#     # callback = CustomCallback()
#     # model.learn(total_timesteps=250_000, callback=callback)
#     # model.save("checkpoint1")
#     # print("reached checkpoint1 \n\n\n\n\n")
#     # plt.plot(smooth_trace(callback.rewards, 100))
#     # plt.draw()
#     # time.sleep(20)
#     # plt.clf()

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


# def train() -> Dict:
#     """
#     TODO: Write this function to train your algorithm.

#     Returns:
#     """
#     raise NotImplementedError("You need to implement this function!")


# def choose_move(state: Any, user_file: Any, verbose: bool = False) -> int:
#     """Called during competitive play. It acts greedily given current state of the board and value
#     function dictionary. It returns a single move to play.

#     Args:
#         state:

#     Returns:
#     """
#     raise NotImplementedError("You need to implement this function!")


def n_games():

    for _ in tqdm(range(100)):
        play_go(
            your_choose_move=choose_move_randomly,
            opponent_choose_move=choose_move_randomly,
            game_speed_multiplier=0.1,
            render=True,
            verbose=True,
        )
        time.sleep(1000)


if __name__ == "__main__":

    ## Example workflow, feel free to edit this! ###
    # file = train()
    # save_pkl(file, TEAM_NAME)

    # check_submission(
    #     TEAM_NAME
    # )  # <---- Make sure I pass! Or your solution will not work in the tournament!!

    # my_network = load_pkl(TEAM_NAME)

    n_games()
    # # Code below plays a single game against a random
    # #  opponent, think about how you might want to adapt this to
    # #  test the performance of your algorithm.
    # def choose_move_no_network(state: Any) -> int:
    #     """The arguments in play_game() require functions that only take the state as input.

    #     This converts choose_move() to that format.
    #     """
    #     return choose_move(state, my_network)

    # play_go(
    #     your_choose_move=choose_move_randomly,
    #     opponent_choose_move=choose_move_randomly,
    #     game_speed_multiplier=1,
    #     render=True,
    #     verbose=False,
    # )
