# region Load Madule
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn import *


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report



# endregiongit branch

# endregion


def load_data():
    # columns
    columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    # read files
    df = pd.read_csv('data/car.data', names=columns, header=None)
    return df