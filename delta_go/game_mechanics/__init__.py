import sys
from pathlib import Path

from .go_base import all_legal_moves
from .go_env import GoEnv, choose_move_randomly, human_player, load_pkl, play_go, save_pkl
from .state import State
from .utils import BLACK, BOARD_SIZE, WHITE

HERE = Path(__file__).parent.parent.parent.resolve()  # isort: ignore
sys.path.append(str(HERE))  # isort: ignore
sys.path.append(str(HERE / "delta_go"))  # isort: ignore
