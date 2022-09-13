from typing import Any, Dict, Optional

import numpy as np
from torch import nn

from check_submission import check_submission
from game_mechanics import GoEnv, choose_move_randomly, load_pkl, play_go, save_pkl

TEAM_NAME = "Team Name"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"


def train() -> Any:
    """
    TODO: Write this function to train your algorithm.

    Returns:
        A pickleable obect to be made available to choose_move
    """
    raise NotImplementedError("You need to implement this function!")


def choose_move(observation: np.ndarray, legal_moves: np.ndarray, pkl_file: Any, env) -> int:
    """Called during competitive play. It acts greedily given current state of the board and value
    function dictionary. It returns a single move to play.

    Args:
        observation: The current stones on the of the board
        legal_moves: The legal moves available on this turn.

    Returns:
        The action to take
    """
    raise NotImplementedError("You need to implement this function!")


if __name__ == "__main__":

    ## Example workflow, feel free to edit this! ###
    file = train()
    save_pkl(file, TEAM_NAME)

    my_network = load_pkl(TEAM_NAME)

    # Code below plays a single game against a random
    #  opponent, think about how you might want to adapt this to
    #  test the performance of your algorithm.
    def choose_move_no_network(observation: np.ndarray, legal_moves: np.ndarray, env) -> int:
        """The arguments in play_game() require functions that only take the state as input.

        This converts choose_move() to that format.
        """
        return choose_move(observation, legal_moves, my_network, env)

    check_submission(
        TEAM_NAME, choose_move_no_network
    )  # <---- Make sure I pass! Or your solution will not work in the tournament!!

    play_go(
        your_choose_move=choose_move_no_network,
        opponent_choose_move=choose_move_randomly,
        game_speed_multiplier=1,
        render=True,
        verbose=True,
    )
