from pathlib import Path
import sys

# Ensure repository-root packages are importable under GitHub Actions and local pytest.
ROOT = Path(__file__).resolve().parents[1]
root_text = str(ROOT)
if root_text not in sys.path:
    sys.path.insert(0, root_text)
