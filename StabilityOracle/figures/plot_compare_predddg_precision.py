from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

import random
from scipy.stats import multivariate_normal
from scipy.stats import wasserstein_distance
from scipy.stats import pearsonr
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties as FP
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import pickle


from sklearn.manifold import TSNE

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import seaborn as sns
import json

sns.set_style("white")
from summary_so import *

lw = 3
fontsize = 12
markersize = 16
markerfacecolor = "None"
markeredgewidth = 4

matplotlib.rc("xtick", labelsize=fontsize)
matplotlib.rc("ytick", labelsize=fontsize)

bar_width = 0.3
polar = [0.71, 0.85, 0.88, 0.95]
hydrophobic = [0.74, 0.85, 0.89, 0.93]
tick_label = [
    "Pred ddG < - .5",
    "Pred ddG < - 1.0",
    "Pred ddG < - 2.0",
    "Pred ddG < - 3.0",
]
x = np.arange(len(polar))

plt.figure(figsize=(10, 6))
plt.bar(x, polar, bar_width, color="#FFADFE", label="polar precision")
plt.bar(
    x + bar_width,
    hydrophobic,
    bar_width,
    color="#69B334",
    label="hydrophobic precision",
)
plt.yticks([0.1, 0.3, 0.5, 0.7, 0.9], size=fontsize)
plt.xticks(x + bar_width, tick_label, size=fontsize + 8, rotation=20)
plt.ylim(0.0, 1.0)
plt.xlim(0 - 0.25, 3 + 0.5)
font = FontProperties(size=fontsize)
plt.legend(prop=font, loc=0)
plt.savefig("hist_pred_ddg.pdf", bbox_inches="tight", pad_inches=0)
plt.close()
