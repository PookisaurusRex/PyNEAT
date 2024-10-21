  
"""Commonly used functions not available in the Python2 standard library."""
from __future__ import division

import numpy as np
from math import sqrt, exp

def mean(values):
    values = list(values)
    return sum(map(float, values)) / len(values)

# Lookup table for commonly used {value} -> value functions.
StatisticalFunctions = { "min": np.nanmin, "max": np.nanmax, "mean": np.nanmean, "median": np.nanmedian, "stdev": np.nanstd }