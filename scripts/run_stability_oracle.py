from pathlib import Path
from typing import Literal
import math
import pickle
import json
import matplotlib.pyplot as plt
import argparse
import logging


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.utils import ModelEma
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable

from StabilityOracle.model import StabilityOraclePipeline


logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def cli():
    parser = argparse.ArgumentParser("Script to run Stability Oracle inference")
    parser.add_argument("--debug", default="", type=str)

    # data
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--outdir", default=Path.cwd() / "predictions", type=Path)
    parser.add_argument("--outfile", default=None, type=Path)

    # model
    parser.add_argument("--use-cls-loss", action="store_true")
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument(
        "--model-ckpt",
        required=True,
        type=Path,
    )
    parser.add_argument("--depth", default=4, type=int)
    parser.add_argument("--use-seq-info", action="store_true")
    parser.add_argument("--use-clstoken", action="store_true")
    parser.add_argument("--clstoken-unify", action="store_true")
    parser.add_argument("--return-idx", default=[3], type=list)
    parser.add_argument("--encoder-decoder", action="store_true")
    parser.add_argument("--out-prod", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")

    # optimizer
    parser.add_argument("--lr", default=0, type=float)
    parser.add_argument("--lr-backbone", default=0, type=float)
    parser.add_argument("--weight-decay", default=0, type=float)

    # training
    parser.add_argument(
        "--batch-size", default=60, type=int, help="total batch size across all GPUs"
    )
    parser.add_argument("--num-iters", default=5000, type=int)
    parser.add_argument("--eval-period", default=1, type=int)

    # logging
    parser.add_argument("--enable-wandb", action="store_true")
    parser.add_argument("--wandb-group", default="stability-oracle", type=str)

    parser.add_argument("--cpu", action="store_true", help="Run inference on CPU; else it checks for GPU or MPS devices.")

    args = parser.parse_args()

    args.outdir.mkdir(0o770, parents=True, exist_ok=True)

    if args.outfile is None:
        args.outfile = args.outdir / f"{args.dataset.stem}.csv"

    assert args.model_ckpt.suffix == ".pt"
    assert args.dataset.suffix == ".jsonl", args.dataset
    assert args.outfile.suffix == ".csv", args.outfile

    return args


if __name__ == "__main__":
    args = cli()
    args.return_idx = [int(item) for item in args.return_idx]
    logging.info(args)

    if args.enable_wandb:
        import wandb

        run_name = args.dataset.stem
        wandb.init(
            project="proteins",
            name=run_name,
            group=args.wandb_group,
            config=args,
            dir=args.outdir,
        )

    pl = StabilityOraclePipeline(args)
    pl.configure_model(mode="regression")
    pl.evaluate()

    if args.enable_wandb:
        wandb.log(score)
        wandb.finish()
