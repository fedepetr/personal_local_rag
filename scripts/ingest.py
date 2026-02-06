from pathlib import Path
import sys

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.pipeline import ingest_folder  # noqa: E402
from rag.settings import DOCS_DIR  # noqa: E402


def run_ingest() -> None:
    ingest_folder(DOCS_DIR)
