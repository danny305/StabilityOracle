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


def extract_ddg(file_path):
    forward_ddgs = []
    permutation_ddgs = []
    with open(file_path, "rb") as f:
        data = list(f)
    count = 0
    for item in data:
        try:
            instance = json.loads(item.strip())
        except:
            continue

        count += 1
        if "ddg" in instance["snapshot"].keys():
            ddg = instance["snapshot"]["ddg"]
        else:
            ddg = instance["snapshot"]["ddG_ML"]

    reverse_ddgs = -np.array(forward_ddgs)
    forward_ddgs = np.array(forward_ddgs)
    permutation_ddgs = np.array(permutation_ddgs)
    print(forward_ddgs.shape, permutation_ddgs.shape)
    return {
        "reverse_ddgs": reverse_ddgs,
        "forward_ddgs": forward_ddgs,
        "permutation_ddgs": permutation_ddgs,
    }


def plot_kde(data, label="t2837"):
    fontsize = 32
    fig, ax1 = plt.subplots(figsize=(12, 6))

    sns.kdeplot(data["reverse_ddgs"], fill=True, label="TR")
    sns.kdeplot(data["forward_ddgs"], fill=True, label="Orig")
    sns.kdeplot(data["permutation_ddgs"], fill=True, label="TP")
    plt.text(-6.5, 0.3, label, ha="center", va="center", fontsize=fontsize + 8)

    plt.xlim(-10, 10)
    plt.ylim(0.0, 0.9)
    sns.despine()
    plt.xticks([-10, -5, 0, 5, 10], size=fontsize + 8)
    plt.yticks([0, 0.2, 0.4, 0.6], size=fontsize + 8)
    plt.xlabel(r"$\Delta\Delta$G", size=fontsize + 8)
    plt.ylabel("Density", size=fontsize + 8)
    plt.legend(fontsize=fontsize)
    plt.savefig(label + "_ddg_distribution.pdf", bbox_inches="tight", pad_inches=0)
    plt.close()
