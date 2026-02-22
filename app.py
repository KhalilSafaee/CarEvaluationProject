# region Load Madule
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import *

# endregion

def load_data():
    # columns
    columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    # read files
    df = pd.read_csv('data/car.data', names=columns, header=None)
    return df