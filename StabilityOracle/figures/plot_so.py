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
datasets = list(prostata_dict.keys())
# datasets = list(score_dict.keys())

# revere rmse
for dataset in score_dict.keys():
    score_dict[dataset]["rmse"] = 1.0 / score_dict[dataset]["rmse"]
for dataset in prostata_dict.keys():
    prostata_dict[dataset]["rmse"] = 1.0 / prostata_dict[dataset]["rmse"]
for dataset in rasp_dict.keys():
    rasp_dict[dataset]["rmse"] = 1.0 / rasp_dict[dataset]["rmse"]

# normalize the metric
maximum_o = defaultdict(int)
minimum_o = defaultdict(lambda: 100)
for dataset in datasets:
    for metric in score_dict[dataset].keys():
        maximum_o[metric] = max(score_dict[dataset][metric], maximum_o[metric])
        minimum_o[metric] = min(score_dict[dataset][metric], minimum_o[metric])
    for metric in prostata_dict[dataset].keys():
        maximum_o[metric] = max(prostata_dict[dataset][metric], maximum_o[metric])
        minimum_o[metric] = min(prostata_dict[dataset][metric], minimum_o[metric])
    for metric in rasp_dict[dataset].keys():
        maximum_o[metric] = max(rasp_dict[dataset][metric], maximum_o[metric])
        minimum_o[metric] = min(rasp_dict[dataset][metric], minimum_o[metric])
magn = 0.2
for metric in score_dict[dataset].keys():
    maximum_o[metric] += magn
    minimum_o[metric] -= magn
colors = {"our": "#0B5085", "prostata": "#8A5300", "rasp": "black"}


for dataset in datasets:

    plt.figure(figsize=(10, 6))
    labels = list(score_dict[dataset].keys())

    orignal_stats = [score_dict[dataset][label] for label in labels]
    orignal_prostata = [prostata_dict[dataset][label] for label in labels]
    orignal_rasp = [rasp_dict[dataset][label] for label in labels]

    stats = orignal_stats
    rasp = orignal_rasp
    prostata = orignal_prostata

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)

    orignal_stats = np.concatenate((orignal_stats, [orignal_stats[0]]))
    orignal_prostata = np.concatenate((orignal_prostata, [orignal_prostata[0]]))
    orignal_rasp = np.concatenate((orignal_rasp, [orignal_rasp[0]]))

    prostata = np.concatenate((prostata, [prostata[0]]))
    rasp = np.concatenate((rasp, [rasp[0]]))
    stats = np.concatenate((stats, [stats[0]]))

    angles = np.concatenate((angles, [angles[0]]))

    plt.polar(
        angles,
        prostata,
        "*-",
        linewidth=lw,
        c=colors["prostata"],
        markersize=markersize,
        markerfacecolor="white",
        markeredgecolor=colors["prostata"],
        markeredgewidth=markeredgewidth,
    )
    plt.polar(
        angles,
        rasp,
        "^-",
        linewidth=lw,
        c=colors["rasp"],
        markersize=markersize,
        markerfacecolor="white",
        markeredgecolor=colors["rasp"],
        markeredgewidth=markeredgewidth,
    )
    plt.polar(
        angles,
        stats,
        "o-",
        linewidth=lw,
        c=colors["our"],
        markersize=markersize,
        markerfacecolor="white",
        markeredgecolor=colors["our"],
        markeredgewidth=markeredgewidth,
    )

    plt.fill(angles, stats, alpha=0.25, c=colors["our"])
    plt.fill(angles, prostata, alpha=0.25, c=colors["prostata"])
    plt.fill(angles, prostata, alpha=0.25, c=colors["rasp"])
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

    if "aug" in dataset:
        shift = 0.08
    else:
        shift = 0.12
    for a, b, c in zip(angles, stats, orignal_stats):
        plt.text(
            a - shift,
            b + shift,
            f"{c:.2f}",
            ha="center",
            va="center",
            fontsize=16,
            c=colors["our"],
        )
    for a, b, c in zip(angles, rasp, orignal_rasp):
        plt.text(
            a + shift,
            b - shift,
            f"{c:.2f}",
            ha="center",
            va="center",
            fontsize=16,
            c=colors["rasp"],
        )

    labels = np.concatenate([labels, [labels[0]]])
    print(prostata.min() - 0.05, stats.max() + 0.05)

    if "aug" in dataset:
        plt.ylim(0.25, 0.95)
    else:
        plt.ylim(0.25, 0.95)
    plt.tick_params("y", labelleft=False)
    plt.thetagrids(angles * 180 / np.pi, labels, FontProperties=font)
    plt.savefig("radar_" + dataset + ".pdf", bbox_inches="tight", pad_inches=0)

### draw lenged ###
import pylab

fig = pylab.figure(facecolor="white")
legend_fig = pylab.figure(facecolor="white")
# sns.set_style("white")
# plot labels
markers = {"our": "o", "prostata": "*", "rasp": "^"}
labels = {"our": "Stability Oracle", "rasp": "RaSP", "prostata": "Prostata-IFML"}
for name in colors.keys():
    fig.gca().plot(
        range(10),
        pylab.randn(10),
        marker=markers[name],
        label=labels[name],
        c=colors[name],
        markersize=markersize,
        markerfacecolor="white",
        markeredgecolor=colors[name],
        markeredgewidth=markeredgewidth,
    )
legend = pylab.figlegend(*fig.gca().get_legend_handles_labels(), loc="center")
# legend.get_frame().set_color('0.90')
legend_fig.canvas.draw()
legend_fig.savefig(
    "radar_legend.pdf",
    facecolor=fig.get_facecolor(),
    bbox_inches=legend.get_window_extent().transformed(
        legend_fig.dpi_scale_trans.inverted()
    ),
)
plt.close()
