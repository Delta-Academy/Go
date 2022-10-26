from typing import Any, Dict, Optional

import numpy as np
import torch

from check_submission import check_submission
from game_mechanics import choose_move_randomly, human_player, load_pkl, play_go, save_pkl
from go_base import all_legal_moves
from state import State

TEAM_NAME = "Team Name"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"


class MCTS:
    def __init__(self):
        """You can use this as an mcts class that persists across choose_move calls."""
        pass

    def prune(self):
        pass

    def set_initial_state(self, state):
        pass


def train() -> Any:
    """
    TODO: Write this function to train your algorithm.

    Returns:
        A pickleable object to be made available to choose_move
    """
    raise NotImplementedError("You need to implement this function!")


def choose_move(
    state: State,
    pkl_file: Optional[Any] = None,
    mcts: Optional[MCTS] = None,
) -> int:
    """Called during competitive play. It returns a single action to play.

    Args:
        state: The current state of the go board (see state.py)
        pkl_file: The pickleable object you returned in train
        env: The current environment

    Returns:
        The action to take
    """
    legal_moves = all_legal_moves(state.board, state.ko)
    raise NotImplementedError("You need to implement this function!")


if __name__ == "__main__":

    # Example workflow, feel free to edit this! ###
    file = train()
    save_pkl(file, TEAM_NAME)

    my_pkl_file = load_pkl(TEAM_NAME)
    my_mcts = MCTS()

    # Choose move functions when called in the game_mechanics expect only a state
    # argument, here is an example of how you can pass a pkl file and an initialized
    # mcts tree
    def choose_move_no_network(state: State) -> int:
        """The arguments in play_game() require functions that only take the state as input.

        This converts choose_move() to that format.
        """
        return choose_move(state, my_pkl_file, mcts=my_mcts)

    # Currently the check submission will fail because we are passing the env to
    # choose_move. We shouldn't do this going forward so am not going to worry
    # about fixing.
    check_submission(
        TEAM_NAME, choose_move_no_network
    )  # <---- Make sure I pass! Or your solution will not work in the tournament!!

    # Play a game against against your bot!
    play_go(
        your_choose_move=human_player,
        opponent_choose_move=choose_move_no_network,
        game_speed_multiplier=1,
        render=True,
        verbose=True,
    )
