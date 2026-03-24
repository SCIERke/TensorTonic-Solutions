import numpy as np
import math

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """

    x = np.array(x)
    return 1.0 / (1.0 + pow(math.e ,-x))
    
