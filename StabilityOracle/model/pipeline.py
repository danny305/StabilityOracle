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

from StabilityOracle.model import SiameseGraphormer


_amino_acids = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLU": 5,
    "GLN": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
}

_reverse_amino_acids = {}
for key in _amino_acids.keys():
    _reverse_amino_acids[_amino_acids[key]] = key


class StabilityOraclePipeline:

    def __init__(self, args):
        self.configured_dataset = False
        self.args = args

    def configure_model(
        self,
        mode: Literal["classification", "regression"],
        hidden_dimension: int = 128,
        num_layers: int = 1,
        drop_rate: float = 0.0,
        **kwargs,
    ) -> None:
        self.mode = mode

        self.hid_dim = hidden_dimension
        self.num_layers = num_layers
        self.drop_rate = drop_rate

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        if self.args.cpu is True:
            self.device = torch.device("cpu")

        logging.info(f"Using device: {self.device}")

        model = SiameseGraphormer(self.args)
        with open(self.args.model_ckpt, "rb") as f:
            ckpt = torch.load(f, map_location=self.device)
        new_ckpt = {}

        for key in ckpt.keys():
            new_ckpt[key[len("module.") :]] = ckpt[key]

        model.load_state_dict(new_ckpt)
        if self.args.debug != "one_gpu":
            logging.debug("Parallelizing inference across GPUs")
            model = nn.DataParallel(model)
        else:
            logging.debug("Force use one GPU")

        model.to(self.device)
        self.model = model
        self.model_ema = ModelEma(model, 0.99)
        self.inference_model = self.model_ema.ema

        self.num_class = 1

    def evaluate(self):

        # build dataset
        feats = []
        coords = []
        mask = []
        cas = []
        label = []
        from_aas = []
        to_aas = []
        mut_infos = []
        pdb_codes = []

        with open(self.args.dataset, "rb") as f:
            data = list(f)

        for data_idx in range(len(data)):
            example = json.loads(data[data_idx])

            mut_infos.append(example["mut_info"])
            pdb_codes.append(example["pdb_id"])
            feats.append(example["input"])
            coords.append(example["coords"])
            mask.append(example["mask"])
            cas.append(example["ca"])
            label.append(-example["ddg"])
            from_aas.append(example["from"])
            to_aas.append(example["to"])

        mut_infos, chain_ids = zip(*[info.split("_", 1) for info in mut_infos])

        feats = torch.from_numpy(np.array(feats)).float().to(self.device)

        coords = torch.from_numpy(np.array(coords)).float().to(self.device)
        mask = torch.from_numpy(np.array(mask)).float().to(self.device)
        cas = torch.from_numpy(np.array(cas)).float().to(self.device)
        from_aas = torch.from_numpy(np.array(from_aas)).long().to(self.device)
        to_aas = torch.from_numpy(np.array(to_aas)).long().to(self.device)

        input_aa = torch.concat(
            (from_aas.reshape(-1, 1), to_aas.reshape(-1, 1)), dim=-1
        )

        model_preds = []

        self.model.eval()
        with torch.no_grad():
            for idx in range(feats.shape[0] // self.args.batch_size + 1):
                bidx = self.args.batch_size * idx
                eidx = self.args.batch_size * idx + self.args.batch_size
                if eidx > feats.shape[0]:
                    eidx = feats.shape[0]
                pred, _ = self.model(
                    feats=feats[bidx:eidx],
                    ca=cas[bidx:eidx],
                    coords=coords[bidx:eidx],
                    mask=mask[bidx:eidx],
                    aa_feats=input_aa[bidx:eidx],
                )
                model_preds += pred.cpu().detach().numpy().tolist()

        logging.info(f"pearson corr: {round(pearsonr(model_preds, label)[0], 4)}")
        logging.info(f"spearman corr: {round(spearmanr(model_preds, label)[0], 4)}")
        logging.info(
            f"AUROC: {round(roc_auc_score(np.array(label) > 0.0, 1 / (1 + np.exp(-np.array(model_preds)))),4)}"
        )

        df = pd.DataFrame(
            {
                "pdb_code": pdb_codes,
                "chain_id": chain_ids,
                "mutation": mut_infos,
                "exp_ddG": np.round(-np.array(label), 4),
                "pred_ddG": np.round(-np.array(model_preds), 4),
            }
        )
        df.to_csv(self.args.outfile, index=False)

        logging.info(f"Predictions wrote to {self.args.outfile}")

        return df
