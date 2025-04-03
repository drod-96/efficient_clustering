# Import librairies
import matplotlib.pyplot as plt
from cycler import cycler

import os

def set_rcParams():
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = "serif"
    plt.rcParams['font.serif'] = ['Computer Modern']
    plt.rcParams['lines.linewidth'] = 1.4
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.prop_cycle'] = cycler(color=['brown', 'b', 'y', 'darkgray'])
    plt.rcParams['legend.labelcolor'] = 'black'
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['xtick.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['ytick.labelcolor'] = 'black'
    plt.rcParams['ytick.color'] = 'black'