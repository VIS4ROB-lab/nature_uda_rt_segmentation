"""
Timing utilities for measuring the time taken by inference.

Usage:
    from uda_helpers.timing import timing, time_statistics

    @timing
    def single_inference(model, img):
        # the original inference code
        ...

    for image in images:
        img = ... (Load image)
        single_inference(model, img)

    time_statistics(single_inference.times)
"""

# ---------------------------------------------------------------
# Copyright (c) 2024 ETH Zurich & UCY V4RL. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


import time
from typing import List, Dict

import numpy as np


def timing(func):
    times = []

    def wrapper(*args, **kwargs):
        start_time = time.time_ns()
        result = func(*args, **kwargs)
        end_time = time.time_ns()
        times.append(end_time - start_time)
        return result

    wrapper.times = times
    return wrapper


def time_statistics(times: List[float]) -> Dict[str, float]:
    times = np.array(times) / 1e9  # convert to seconds
    return {
        'min': np.min(times),
        'max': np.max(times),
        'mean': np.mean(times),
        'std': np.std(times),
    }
