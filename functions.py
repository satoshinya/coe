import numpy as np

def weighted_average(scores, weights):
    wsum = np.sum(weights)
    if len(scores) > 0 and wsum > 0:
        return np.sum(scores * weights) / wsum
    else:
        return 0.0
