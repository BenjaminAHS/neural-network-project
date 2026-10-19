"""Rebuild the notebooks from their .py sources (jupytext) and run them from top to bottom.

    python scripts/run_notebooks.py            # all notebooks, in order
    python scripts/run_notebooks.py 01 02      # only the notebooks whose name starts with 01 or 02

Set FAST_RUN=1 for a quick check (short trainings) and SHOW_ARTWORKS=1 to see the paintings locally
(never commit a notebook run with SHOW_ARTWORKS=1: it would contain images of the artworks).
"""
import subprocess
import sys
import time
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


def run(source):
    t0 = time.perf_counter()
    subprocess.run([sys.executable, "-m", "jupytext", "--to", "ipynb", source.name], cwd=NOTEBOOKS, check=True)
    # record_timing=False: no execution timestamps, so a re-run only changes the notebook if a result changes
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                    "--ExecutePreprocessor.timeout=-1", "--ExecutePreprocessor.record_timing=False",
                    source.with_suffix(".ipynb").name],
                   cwd=NOTEBOOKS, check=True)
    print(f"{source.stem}: done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    prefixes = sys.argv[1:]
    for source in sorted(NOTEBOOKS.glob("[0-9][0-9]_*.py")):
        if not prefixes or any(source.name.startswith(p) for p in prefixes):
            run(source)
