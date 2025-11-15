import numpy as np
import pandas as pd

def bass_model(t, p, q, M):
    C = np.cumsum(t) 
    return p * (M - C) + q * C * (C / M)

