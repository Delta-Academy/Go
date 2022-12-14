from typing import Any, Optional

from check_submission import check_submission
from game_mechanics import (
    State,
    all_legal_moves,
    choose_move_randomly,
    human_player,
    is_terminal,
    load_pkl,
    play_go,
    reward_function,
    save_pkl,
    transition_function,
)

TEAM_NAME = "Team Name"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"


class MCTS:
    def __init__(self):
        """This class will persist across choose_move() calls, meaning pruning is possible."""
        pass

    def prune_tree(self):
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

    check_submission(
        TEAM_NAME, choose_move_no_network
    )  # <---- Make sure I pass! Or your solution will not work in the tournament!!

    # Play a game against your bot!
    # Left click to place a stone. Right click to pass!
    play_go(
        your_choose_move=human_player,
        opponent_choose_move=choose_move_no_network,
        game_speed_multiplier=1,
        render=True,
        verbose=True,
    )
