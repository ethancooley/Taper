"""Set up Taper: generate the dataset and produce the evaluation figure.

Attribution: This file was written with the assistance of Claude (Anthropic).
"""

import subprocess
import sys


def main() -> None:
    for script in ("scripts/make_dataset.py", "scripts/evaluate.py"):
        subprocess.run([sys.executable, script], check=True)


if __name__ == "__main__":
    main()
