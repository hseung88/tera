import sys
from pathlib import Path as path_type

sys.path.insert(0, str(path_type(__file__).resolve().parents[1] / "src"))

from tera.experiments.bo.plot_cli import main

if __name__ == "__main__":
    main()
