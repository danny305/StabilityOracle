from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

import random
from scipy.stats import multivariate_normal
from scipy.stats import wasserstein_distance
import matplotlib

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties as FP
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr

# import seaborn as sns
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import seaborn as sns
import json

sns.set_style("white")

# glue = sns.load_dataset("glue").pivot("Model", "Task", "Score")
# print(glue.head)
# sns.heatmap(glue)

lw = 3
fontsize = 20
markersize = 36
markerfacecolor = "None"
markeredgewidth = 4

matplotlib.rc("xtick", labelsize=fontsize)
matplotlib.rc("ytick", labelsize=fontsize)

three2one = {
    "CYS": "C",
    "ASP": "D",
    "SER": "S",
    "GLN": "Q",
    "LYS": "K",
    "ILE": "I",
    "PRO": "P",
    "THR": "T",
    "PHE": "F",
    "ASN": "N",
    "GLY": "G",
    "HIS": "H",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "ALA": "A",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
    "ASX": "D",
}
one2three = {}
for key in three2one.keys():
    one2three[three2one[key]] = key

colors = [
    "#CC0000",
    "#0000CC",
    "#CCCC00",
    "#00CCCC",
    "#CC6600",
    "#00CC00",
]
lw = 3
fontsize = 24
markersize = 16
markerfacecolor = "None"
markeredgewidth = 4

matplotlib.rc("xtick", labelsize=fontsize)
matplotlib.rc("ytick", labelsize=fontsize)


acc = [0.5584, 0.6337]
recall = [0.5792, 0.6225]
precision = [0.4859, 0.5568]
x_val = np.array([0, 1])
width = 0.2

fig, ax1 = plt.subplots(figsize=(10, 6))
plt.bar(x_val, acc, width, label="Accuracy", color=colors[0], alpha=0.75)
plt.bar(x_val + width, recall, width, label="Recall", color=colors[1], alpha=0.75)
plt.bar(
    x_val + 2 * width, precision, width, label="Precision", color=colors[2], alpha=0.75
)
for idx, metric in enumerate([acc, recall, precision]):
    for _, item in enumerate(metric):
        plt.text(
            _ + width * idx,
            metric[_] + 0.02,
            f"{metric[_]:.2f}",
            ha="center",
            va="center",
            fontsize=24,
            c=colors[idx],
        )

plt.xlim(-0.05, 1.65)
plt.ylim(0.2, 0.8)
sns.despine()
plt.xticks([0.2, 1.2], ["C2878 Filter", "C2878 Filter + TP"], size=fontsize)
plt.yticks([0.2, 0.5, 0.8], size=fontsize)
plt.xlabel(r"Training Dataset", size=fontsize)
# plt.legend(fontsize=fontsize)
plt.savefig("c28_filter_permutation.pdf", bbox_inches="tight", pad_inches=0)
plt.close()


acc = [0.5185, 0.6296]
recall = [0.2800, 0.4000]
precision = [0.4667, 0.6667]
width = 0.2


fig, ax1 = plt.subplots(figsize=(10, 6))
plt.bar(x_val, acc, width, label="Accuracy", color=colors[0], alpha=0.75)
plt.bar(x_val + width, recall, width, label="Recall", color=colors[1], alpha=0.75)
plt.bar(
    x_val + 2 * width, precision, width, label="Precision", color=colors[2], alpha=0.75
)

for idx, metric in enumerate([acc, recall, precision]):
    for _, item in enumerate(metric):
        plt.text(
            _ + width * idx,
            metric[_] + 0.02,
            f"{metric[_]:.2f}",
            ha="center",
            va="center",
            fontsize=24,
            c=colors[idx],
        )

plt.xlim(-0.05, 1.65)
plt.ylim(0.2, 0.8)
sns.despine()
plt.xticks([0.2, 1.2], ["C2878", "C2878 + TP"], size=fontsize)
plt.yticks([0.2, 0.5, 0.8], size=fontsize)
plt.xlabel(r"Training Dataset", size=fontsize)
plt.legend(fontsize=fontsize)
plt.savefig("c28_permutation.pdf", bbox_inches="tight", pad_inches=0)
plt.close()
