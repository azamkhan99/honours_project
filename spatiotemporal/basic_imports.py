# Basic imports I use in almost all notebooks.
# Plotting
import seaborn as sns
import matplotlib.pyplot as plt

#import pandas as pd
import numpy as np
import os
from datetime import timedelta

# Register matplotlib as datetime converter
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from collections import Counter
