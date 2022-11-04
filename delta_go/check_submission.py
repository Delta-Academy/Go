import datetime
import hashlib
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union

import numpy as np

from game_mechanics import BOARD_SIZE, GoEnv, choose_move_randomly, load_pkl
from torch import nn


def check_submission(team_name: str, choose_move_no_network: Callable) -> None:
    example_state, _, _, info = GoEnv(choose_move_randomly).reset()
    expected_choose_move_return_type = (int, np.int64)
    expected_pkl_output_type = Any
    pkl_file = load_pkl(team_name)

    max_time = 5
    t1 = time.time()
    choose_move_no_network(
        example_state,
    )
    t2 = time.time()
    assert t2 - t1 < max_time

    return _check_submission(
        example_state=example_state,
        expected_choose_move_return_type=expected_choose_move_return_type,
        expected_pkl_type=None,
        pkl_file=pkl_file,
        pkl_checker_function=lambda x: x,
        current_folder=Path(__file__).parent.resolve(),
        choose_move_extra_argument={"legal_moves": np.arange(BOARD_SIZE**2 + 1)},
    )


HERE = Path(__file__).parent.resolve()


def hash_game_mechanics(path: Path) -> str:
    """Call me to generate game_mechanics_hash."""
    return sha256_file(path / "game_mechanics.py")


def get_local_imports(folder_path) -> Set:
    """Get the names of all files imported from folder_path."""
    local_imports = set()
    for module in sys.modules.values():
        if not hasattr(module, "__file__") or module.__file__ is None:
            continue
        path = Path(module.__file__)
        # Module is in this folder
        if path.parent == folder_path:
            local_imports.add(path.stem)
    return local_imports


def _check_submission(
    example_state: Any,
    expected_choose_move_return_type: Union[Type, Tuple[Type, ...]],
    current_folder: Path,
    pkl_file: Optional[Any] = None,
    expected_pkl_type: Optional[Type] = None,
    pkl_checker_function: Optional[Callable] = None,
    choose_move_extra_argument: Optional[Dict[str, Any]] = None,
) -> None:
    """Checks a user submission is valid.

    Args:
        example_state (any): Example of the argument to the user's choose_move function
        expected_choose_move_return_type (Type): of the users choose_move_function
        game_mechanics_hash (str): sha256 hash of game_mechanics.py (see hash_game_mechanics())
        current_folder (Path): The folder path of the user's game code (main.py etc)
        pkl_file (any): The user's loaded pkl file (None if not using a stored pkl file)
        expected_pkl_type (Type): Expected type of the above (None if not using a stored pkl file)
        pkl_checker_function (callable): The function to check that pkl_file is valid
                                         (None if not using a stored pkl file)
    """

    local_imports = get_local_imports(current_folder)
    valid_local_imports = {
        "__main__",
        "__init__",
        "game_mechanics",
        "check_submission",
        "main",
        "utils",
        "models",
    }
    if not local_imports.issubset(valid_local_imports):
        warnings.warn(
            f"You imported {local_imports - valid_local_imports}. "
            f"Please do not import local files other than "
            f"check_submission and game_mechanics into your main.py."
        )

    main = current_folder / "main.py"
    assert main.exists(), "You need a main.py file!"
    assert main.is_file(), "main.py isn't a Python file!"

    file_name = main.stem

    pre_import_time = datetime.datetime.now()
    mod = __import__(f"{file_name}", fromlist=["None"])
    time_to_import = (datetime.datetime.now() - pre_import_time).total_seconds()

    # Check importing takes a reasonable amount of time
    assert time_to_import < 2, (
        f"Your main.py file took {time_to_import} seconds to import.\n"
        f"This is much longer than expected.\n"
        f"Please make sure it's not running anything (training, testing etc) outside the "
        f"if __name__ == '__main__': at the bottom of the file"
    )

    # Check the choose_move() function exists
    try:
        choose_move = getattr(mod, "choose_move")
    except AttributeError as e:
        raise Exception(f"No function 'choose_move()' found in file {file_name}.py") from e

    # Check there is a TEAM_NAME attribute
    try:
        team_name = getattr(mod, "TEAM_NAME")
    except AttributeError as e:
        raise Exception(f"No TEAM_NAME found in file {file_name}.py") from e

    # Check TEAM_NAME isn't empty

    if len(team_name) == 0:
        raise ValueError(f"TEAM_NAME is empty in file {file_name}.py")

    # Check TEAM_NAME isn't still 'Team Name'
    if team_name == "Team Name":
        raise ValueError(
            f"TEAM_NAME='Team Name' which is what it starts as - "
            f"please change this in file {file_name}.py to your team name!"
        )

    congrats_str = "Congratulations! Your Repl is ready to submit :)"
    if pkl_file is not None:
        congrats_str += f"It'll be using value function file called 'dict_{team_name}.pkl'"
    print(congrats_str)


def sha256_file(filename: Path) -> str:
    """stackoverflow.com/a/44873382."""
    hasher = hashlib.sha256()
    # Create a memory buffer
    buffer = bytearray(128 * 1024)
    mv = memoryview(buffer)
    with open(filename, "rb", buffering=0) as f:
        # Read the file into the buffer, chunk by chunk
        while chunk := f.readinto(mv):  # type: ignore
            hasher.update(mv[:chunk])
    # Hash the complete file
    return hasher.hexdigest()


def pkl_checker_value_dict(pkl_file: Dict) -> None:
    """Checks a dictionary acting as a value lookup table."""
    if isinstance(pkl_file, defaultdict):
        assert not callable(
            pkl_file.default_factory
        ), "Please don't use functions within default dictionaries in your pickle file!"

    assert len(pkl_file) > 0, "Your dictionary is empty!"

    for k, v in pkl_file.items():
        assert isinstance(
            v, (float, int)
        ), f"Your value function dictionary values should all be numbers, but for key {k}, the value {v} is of type {type(v)}!"
