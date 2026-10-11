from pathlib import Path as path_type

import pandas as pd

from tera.methods.names import normalize_method_name


def normalize_results(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "method" in frame:
        frame["method"] = frame["method"].map(normalize_method_name)
    return frame


def read_results(path: str | path_type) -> pd.DataFrame:
    return normalize_results(pd.read_csv(path))
