import pandas as pd
import numpy as np

# Define the functions used in the equation
def emwa(s, span):
    res = pd.DataFrame(s).ewm(span=span, min_periods=span).mean()
    return res[0].values

def sigmoid(s):
    return 1 / (1 + np.exp(-s))

def exp(s):
    return np.exp(s)