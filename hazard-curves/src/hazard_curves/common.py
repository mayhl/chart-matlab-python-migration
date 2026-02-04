
from dataclasses import fields

import numpy as np
from numpy.typing import NDArray


def aef2aep(in_values):
    """
    By: E. Ramos-Santiago
    Description: Script to adjust the trend of hazard curves with jumps. The
    StormSim-SST tool can produce non-monotonic curves when the GPD threshold
    parameter returned by the MRL selection method is too low. This causes
    incomplete random samples and the potential to have jumps in the mean
    curve and CLs.
    History of revisions:
    20210310-ERS: created function to patch the SST tool.
    """
    return (np.exp(in_values) - 1) / np.exp(in_values)


def bool_check(checks):

    def _bool_check(value, message, allowed):
        if value is None:
            raise ValueError(message)
        if not np.isscalar(value):
            raise ValueError(message)
        if value not in allowed:
            raise ValueError(message)

    for obj, name, allowed, msg in checks:
        _bool_check(getattr(obj, name), f"{name} must be {msg}", allowed)

def get_grid_values(is_use_aep: bool) -> tuple[NDArray, NDArray, NDArray]:
    #fmt: off
    if is_use_aep:
        tbl_x = 1. / np.array([2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6])
    else:
        #tbl_x = 1. / np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000])
        tbl_x = 1. / np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6])
    # fmt: on

    rsp_y = np.arange(0.01, 20.01, 0.01)

    # AEFs for plotting (log-spaced, MATLAB-equivalent)
    d = 1 / 90
    v = 10 ** np.arange(1, -d, -d)
    plt_x = v.copy()
    x = 10.0
    for _ in range(6):
        plt_x = np.concatenate([plt_x, v[1:] / x])
        x *= 10
    plt_x = plt_x[::-1]

    return tbl_x, rsp_y, plt_x


def dict_to_dataclass(cls, dict_obj):
    """Convert a dictionary to a dataclass instance."""
    if not hasattr(cls, "__dataclass_fields__"):
        raise ValueError("cls must be a dataclass type.")
    valid_keys = {f.name for f in fields(cls)}
    filtered_dict = {k: v for k, v in dict_obj.items() if k in valid_keys}
    return cls(**filtered_dict)


