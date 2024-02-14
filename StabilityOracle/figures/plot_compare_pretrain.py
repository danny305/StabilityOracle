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
from matplotlib.font_manager import FontProperties
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
from summary_so import *

# glue = sns.load_dataset("glue").pivot("Model", "Task", "Score")
# print(glue.head)
# sns.heatmap(glue)

lw = 3
fontsize = 16
markersize = 16
markerfacecolor = "None"
markeredgewidth = 4

matplotlib.rc("xtick", labelsize=fontsize)
matplotlib.rc("ytick", labelsize=fontsize)


# rank due to pearson value
datasets = list(c28scratch_dict.keys())

# revere rmse
dicts = {
    "orig": c28orig_dict,
    "scratch": c28scratch_dict,
    "aug": c28aug_dict,
    "scratch_orig": c28origscratch_dict,
}
for dataset in datasets:
    for key in dicts.keys():
        dicts[key][dataset]["rmse"] = 1.0 / dicts[key][dataset]["rmse"]

colors = {
    "orig": "#FF590E",
    "scratch": "#260FF5",
    "aug": "#00FB78",
    "scratch_orig": "#F5D307",
}
markers = {"orig": "o", "scratch": "^", "aug": "*", "scratch_orig": "<"}
for dataset in datasets:
    plt.figure(figsize=(10, 6))
    labels = list(score_dict[dataset].keys())

    # normalize the metric
    maximum = defaultdict(int)
    minimum = defaultdict(lambda: 100)
    for key in dicts.keys():
        score_dict = dicts[key]
        for metric in score_dict[dataset].keys():
            maximum[metric] = max(score_dict[dataset][metric], maximum[metric])
            minimum[metric] = min(score_dict[dataset][metric], minimum[metric])

    magn = 0.2
    for metric in score_dict[dataset].keys():
        maximum[metric] += magn
        minimum[metric] -= magn

    for key in ["orig", "aug", "scratch_orig", "scratch"]:
        score_dict = dicts[key]
        orignal_stats = [score_dict[dataset][label] for label in labels]
        stats = orignal_stats
        # stats = [( score_dict[dataset][label] - minimum[label]) / (maximum[label] - minimum[label]) for label in labels]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        orignal_stats = np.concatenate((orignal_stats, [orignal_stats[0]]))

        stats = np.concatenate((stats, [stats[0]]))
        angles = np.concatenate((angles, [angles[0]]))

        plt.polar(
            angles,
            stats,
            markers[key] + "-",
            linewidth=lw,
            c=colors[key],
            markersize=markersize,
            markerfacecolor="white",
            markeredgecolor=colors[key],
            markeredgewidth=markeredgewidth,
        )

        plt.fill(angles, stats, alpha=0.25, c=colors[key])

        if "aug" in key:
            for a, b, c in zip(angles, stats, orignal_stats):
                plt.text(
                    a - 0.1,
                    b + 0.1,
                    f"{c:.2f}",
                    ha="center",
                    va="center",
                    fontsize=16,
                    c=colors[key],
                )
        elif "scratch_orig" in key:
            for a, b, c in zip(angles, stats, orignal_stats):
                plt.text(
                    a + 0.1,
                    b - 0.1,
                    f"{c:.2f}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    c=colors[key],
                )

    font = FontProperties(size=fontsize)
    # revere rmse
    for idx in range(len(labels)):
        if labels[idx] == "rmse":
            labels[idx] = "1 / rmse"
        if labels[idx] == "accuracy":
            labels[idx] = "acc"
        if labels[idx] == "precision":
            labels[idx] = "recall"
        elif labels[idx] == "recall":
            labels[idx] = "precision"

    labels = np.concatenate([labels, [labels[0]]])
    plt.ylim(0.1, 0.9)
    plt.tick_params("y", labelleft=False)
    plt.thetagrids(angles * 180 / np.pi, labels, FontProperties=font)
    plt.savefig(
        "compare_pretrain_" + dataset + ".pdf", bbox_inches="tight", pad_inches=0
    )

### draw lenged ###
import pylab

fig = pylab.figure(facecolor="white")
legend_fig = pylab.figure(facecolor="white")
# sns.set_style("white")
# plot labels

labels = {
    "orig": "C2878+TR (SS Pretrained)",
    "scratch": "C2878+TR+TP (no SS Pretraining)",
    "aug": "C2878+TR+TP (SS Pretrained)",
    "scratch_orig": "C2878+TR (no SS Pretraining)",
}
for name in ["scratch_orig", "scratch", "orig", "aug"]:
    fig.gca().plot(
        range(10),
        pylab.randn(10),
        marker=markers[name],
        label=labels[name],
        c=colors[name],
        markersize=markersize - 4,
        markerfacecolor="white",
        markeredgecolor=colors[name],
        markeredgewidth=markeredgewidth,
    )
legend = pylab.figlegend(*fig.gca().get_legend_handles_labels(), loc="center")
# legend.get_frame().set_color('0.90')
legend_fig.canvas.draw()
legend_fig.savefig(
    "compare_legend.pdf",
    facecolor=fig.get_facecolor(),
    bbox_inches=legend.get_window_extent().transformed(
        legend_fig.dpi_scale_trans.inverted()
    ),
)
plt.close()
