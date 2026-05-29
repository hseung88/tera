
from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from gp_sim_kl.cli.plot_gp_sim_kl import main

if __name__ == "__main__":
    main()
