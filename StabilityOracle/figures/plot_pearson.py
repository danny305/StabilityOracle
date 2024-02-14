from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

import random
from scipy.stats import multivariate_normal
from scipy.stats import wasserstein_distance
from scipy.stats import pearsonr
import matplotlib

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties as FP
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import pickle


# method for dimension reduction
from sklearn.manifold import TSNE

# import seaborn as sns
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import seaborn as sns
import json

sns.set_style("white")
# plt.style.use("dark_background")
# sns.set(rc={'figure.facecolor':'gray'})
# sns.set(style="whitegrid", palette="colorblind", color_codes=True)
# plt.style.use('seaborn-colorblind')

# glue = sns.load_dataset("glue").pivot("Model", "Task", "Score")
# print(glue.head)
# sns.heatmap(glue)

from summary_pearson import *

lw = 1
fontsize = 10
markersize = 96
markerfacecolor = "None"
markeredgewidth = 4

matplotlib.rc("xtick", labelsize=fontsize)
matplotlib.rc("ytick", labelsize=fontsize)

fig, ax1 = plt.subplots(figsize=(12, 3))
x_val = 0

# generate label
methods = set()
# rank due to pearson value
datasets = score_dict.keys()
rank_datasets = [
    (dataset, score_dict[dataset]["our"])
    for dataset in datasets
    if "t2837" not in dataset
]
rank_datasets.sort(key=lambda x: x[1])
datasets = [item[0] for item in rank_datasets]
datasets = ["t2837", "t2837_reverse"] + datasets

maximum = defaultdict(lambda: [0, ""])
for dataset in datasets:
    print(dataset)
    for method in score_dict[dataset].keys():
        if method == "porstata" and "p53" in dataset:
            ax1.scatter(
                x_val,
                score_dict[dataset][method],
                marker=mark[method],
                c=colors[method],
                s=int(markersize * 1.8),
            )
        else:
            ax1.scatter(
                x_val,
                score_dict[dataset][method],
                marker=mark[method],
                c=colors[method],
                s=markersize,
            )
        methods.add(method)
        if method != "our":
            if score_dict[dataset][method] > maximum[dataset][0]:
                maximum[dataset] = [score_dict[dataset][method], method]
    method = "our"
    ax1.scatter(
        x_val,
        score_dict[dataset][method],
        marker=mark[method],
        c=colors[method],
        s=markersize,
    )
    plt.text(
        x_val,
        score_dict[dataset][method] + 0.03,
        f"{score_dict[dataset][method]:.2f}",
        ha="center",
        va="center",
        fontsize=8,
        c=colors[method],
    )
    x_val += 1
plt.plot([1.5] * 6, np.arange(6), c="black", linewidth=lw)
x_val = 0
for dataset in datasets:
    score, method = maximum[dataset]
    plt.text(
        x_val + 0.3,
        score,
        f"{score:.2f}",
        ha="center",
        va="center",
        fontsize=8,
        c=colors[method],
    )
    x_val += 1

plt.yticks([0.20, 0.35, 0.50, 0.65], size=fontsize)
plt.xticks(
    list(range(x_val)),
    [" ".join(dataset.split("_")) for dataset in datasets],
    size=fontsize,
    rotation=45,
)
plt.grid(axis="x")
plt.ylim(0.1, 0.80)
plt.xlim(0 - 0.5, x_val - 0.5)
sns.despine()
plt.savefig("pearson.pdf", bbox_inches="tight", pad_inches=0)
plt.close()


### draw lenged ###
import pylab

fig = pylab.figure(facecolor="white")
legend_fig = pylab.figure(facecolor="white")
# sns.set_style("white")
# plot labels
methods.remove("our")
methods = ["our"] + list(methods)
for name in methods:
    fig.gca().scatter(
        range(10),
        pylab.randn(10),
        s=markersize,
        marker=mark[name],
        label=labels[name],
        c=colors[name],
    )
    # ax1.scatter( -1, -1, label=labels[method], marker=mark[method], c=colors[method], s=markersize)
legend = pylab.figlegend(*fig.gca().get_legend_handles_labels(), loc="center")
# legend.get_frame().set_color('0.90')
legend_fig.canvas.draw()
legend_fig.savefig(
    "pearson_legend.pdf",
    facecolor=fig.get_facecolor(),
    bbox_inches=legend.get_window_extent().transformed(
        legend_fig.dpi_scale_trans.inverted()
    ),
)
plt.close()
