# mypy: ignore-errors
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE))

from main import workflow  # isort:skip

workflow()
