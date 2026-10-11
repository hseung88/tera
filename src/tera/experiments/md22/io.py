from tera.methods.names import md22_method_label
from tera.plotting.io import read_results as _read_results


def read_results(path):
    frame = _read_results(path)
    if "method" in frame:
        frame["method"] = frame["method"].map(md22_method_label)
    return frame
