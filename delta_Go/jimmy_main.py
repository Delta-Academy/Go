from typing import Any, Dict

from tqdm import tqdm

from check_submission import check_submission
from game_mechanics import GoEnv, choose_move_randomly, play_go, choose_move_pass

TEAM_NAME = "Team Jimmy"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"


def train() -> Dict:
    """
    TODO: Write this function to train your algorithm.

    Returns:
    """
    raise NotImplementedError("You need to implement this function!")


def choose_move(state: Any, user_file: Any, verbose: bool = False) -> int:
    """Called during competitive play. It acts greedily given current state of the board and value
    function dictionary. It returns a single move to play.

    Args:
        state:

    Returns:
    """
    raise NotImplementedError("You need to implement this function!")


def n_games():

    for _ in tqdm(range(100)):
        play_go(
            your_choose_move=choose_move_randomly,
            opponent_choose_move=choose_move_randomly,
            game_speed_multiplier=0.5,
            render=True,
            verbose=True,
        )


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
