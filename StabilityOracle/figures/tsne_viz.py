from pathlib import Path
import pylab
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

# import matplotlib.colors as mcolors
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import pickle

# method for dimension reduction
from sklearn.manifold import TSNE

# import umap
import umap.umap_ as umap

# import seaborn as sns
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# import seaborn as sns
import json

# sns.set_style("white")

ALL_AAs = "ACDEFGHIKLMNPQRSTVWY"
lw = 3
fontsize = 16
markersize = 8
markerfacecolor = "None"
markeredgewidth = 4

matplotlib.rc("xtick", labelsize=fontsize)
matplotlib.rc("ytick", labelsize=fontsize)

amino_acid_colors = {
    "A": "#FF8C33",  # orange
    "G": "#FFA07F",  # light salmon
    "P": "#FF1493",  # deep pink
    "I": "#00C4C4",  # dark turquoise
    "L": "#00A8FF",  # deep sky blue
    "V": "#1E7FFF",  # dodger blue
    "F": "#8A4AFF",  # blue violet
    "W": "#8F00FF",  # dark violet
    "Y": "#8F35FF",  # dark orchid
    "D": "#FF66B2",  # hot pink
    "N": "#FFB5B5",  # light pink
    "E": "#FFB5B5",  # pink
    "Q": "#FFDEDE",  # misty rose
    "R": "#FF4C4C",  # crimson
    "K": "#FF0000",  # red
    "H": "#FF5733",  # tomato
    "S": "#A8D8E6",  # light blue
    "T": "#73C6FA",  # sky blue
    "C": "#FFFF4D",  # yellow
    "M": "#FFD633",  # gold
}


def plot_to(
    hidden_embedded, to_aa, color_map: dict = amino_acid_colors, title: str = ""
):
    scatters = []
    for aa in ALL_AAs:
        cond = to_aa == aa
        scatters.append(
            plt.scatter(
                hidden_embedded[cond, 0],
                hidden_embedded[cond, 1],
                alpha=1.0,
                c=color_map[aa],
                s=markersize,
                marker="o",
                edgecolors=None,
                label=aa,
            )
        )
    plt.yticks(size=fontsize)
    plt.xticks(size=fontsize)
    plt.savefig(title, bbox_inches="tight", pad_inches=0)
    plt.close()


if __name__ == "__main__":

    colors = {
        "C": "#ffcc33",
        "M": "#ffbb22",
        "I": "#5cff33",
        "L": "#c0ebbd",
        "V": "#0c5507",
        "R": "#3397ff",
        "K": "#3361ff",
        "H": "#57ffd8",
        "N": "#ff9061",
        "E": "#6b0000",
        "D": "#b30000",
        "Q": "#d4612b",
        "A": "#78a96a",
        "G": "#c6ee17",
        "P": "#fd1293",
        "F": "#923ea8",
        "W": "#c68cd4",
        "Y": "#ee17dc",
        "S": "#b07736",
        "T": "#dda769",
    }

    with open("uniq_residue.pkl", "rb") as f:
        data = pickle.load(f)
    to_aa = np.array(data["aa"])
    data = data["feats"]

    plot_to(data, to_aa, color_map=colors, title="umap_cdna_to.pdf")
    plt.close()

    fig = pylab.figure(facecolor="white")
    legend_fig = pylab.figure(facecolor="white")
    markers = {}
    for key in colors.keys():
        markers[key] = "o"
    labels = {}
    for key in colors.keys():
        labels[key] = key
    for name in colors.keys():
        fig.gca().scatter(
            range(10),
            pylab.randn(10),
            marker=markers[name],
            label=labels[name],
            c=colors[name],
            s=markersize,
        )
    legend = pylab.figlegend(
        *fig.gca().get_legend_handles_labels(), loc="center", ncol=2
    )
    legend_fig.canvas.draw()
    legend_fig.savefig(
        "umap_cdna_legend.pdf",
        facecolor=fig.get_facecolor(),
        bbox_inches=legend.get_window_extent().transformed(
            legend_fig.dpi_scale_trans.inverted()
        ),
    )
    plt.close()
