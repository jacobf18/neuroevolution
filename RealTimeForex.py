import MultiNEAT as NEAT
import pandas as pd
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from joblib import Parallel, delayed
import multiprocessing
from backtesting import Backtest, Strategy
import ta
from pandas_datareader import data
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def fitness_function(positions, training_prices):
    