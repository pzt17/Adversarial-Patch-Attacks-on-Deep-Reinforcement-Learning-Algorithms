import numpy as np

def safe_mean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

