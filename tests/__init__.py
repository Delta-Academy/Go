import sys
from pathlib import Path

HERE = Path(__file__).parent.parent.resolve()  # isort: ignore
sys.path.append(str(HERE))  # isort: ignore
sys.path.append(str(HERE / "delta_go"))  # isort: ignore
