"""Make the repo root importable so `import config` and `import fpl_optimiser`
work under pytest and from plain `python` runs started in the repo root."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
