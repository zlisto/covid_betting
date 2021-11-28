import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import *
from datetime import date

__author__ = 'kq'


def get_root() -> str:
    return str(Path(__file__).parent)


# Figures
FIGURE_SIZE = (9, 9)
FIGURES = get_root() + '/figures/{}.png'
ROTATION = '45'
LABEL_SIZE = 10


# Global
COVID = '2020-03-12'
DATE = 'date'
DATE_FORMAT = '%Y-%m-%d'
FILE_NAME = get_root() + '/data/sports_data.csv'
SPORT = 'sport'
START_DATE = '2010-04-04'
COLS = ['date', 'season', 'Team_visitor', 'Team_home', 'Final_visitor', 'Final_home', 'ML_visitor', 'ML_home']
HOME, VISITOR = 'home', 'visitor'
SOURCE = 'https://www.sportsbookreviewsonline.com/scoresoddsarchives/{}/{}%20odds%20{}-{}.xlsx'
START, END = 2007, date.today().year
FILE_NAME = "data/sports_data_new.csv"
FILE_NAME_CLEAN ="data/sports_data_cleaned.csv"
FILE_NAME_PPG = 'data/sports_ppg.csv'
ALL_STAR_BREAK = '2021-03-10'
SCHEMES = ['probability', 
           'inverse_probability',
           'bernoulli',
           'inverse_bernoulli',
           'moneyline',
           'inverse_moneyline']
# Plotting
plt.style.use('seaborn-whitegrid')
