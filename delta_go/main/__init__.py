import sys
from pathlib import Path

from .main import MCTS, TEAM_NAME, choose_move

HERE = Path(__file__).parent.parent.parent.resolve()  # isort: ignore
sys.path.append(str(HERE))  # isort: ignore
sys.path.append(str(HERE / "delta_go"))  # isort: ignore
