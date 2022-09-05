import time
from pathlib import Path
from typing import Callable

import delta_utils.check_submission as checker
import numpy as np
from torch import nn

from game_mechanics import BOARD_SIZE, GoEnv, choose_move_randomly, load_pkl


def check_submission(team_name: str, choose_move_no_network: Callable) -> None:
    example_state, _, _, info = GoEnv(choose_move_randomly).reset()
    expected_choose_move_return_type = (int, np.int64)
    game_mechanics_expected_hash = (
        "c888eaf4e6c970fee4d89bd35cd64b14c33a6ea87bef6fac4fce76da92be7fe1"
    )
    expected_pkl_output_type = nn.Module
    pkl_file = load_pkl(team_name)

    max_time = 5
    t1 = time.time()
    choose_move_no_network(example_state, info["legal_moves"])
    t2 = time.time()
    assert t2 - t1 < max_time

    return checker.check_submission(
        example_state=example_state,
        expected_choose_move_return_type=expected_choose_move_return_type,
        expected_pkl_type=expected_pkl_output_type,
        pkl_file=pkl_file,
        pkl_checker_function=lambda x: x,
        game_mechanics_hash=game_mechanics_expected_hash,
        current_folder=Path(__file__).parent.resolve(),
        choose_move_extra_argument={"legal_moves": np.arange(BOARD_SIZE**2 + 1)},
    )
