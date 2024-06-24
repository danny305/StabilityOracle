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
from StabilityOracle.model.dataloader import load_raw_graph, load_embedded_graph
from StabilityOracle.model.post_process import format_dms_predictions

        
class StabilityOraclePipeline:

    def __init__(self, args):
        self.configured_dataset = False
        self.args = args
        self.dms = False

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
        if self.device != torch.device("cpu"):
            if self.args.debug != "one_gpu":
                logging.info("Parallelizing inference across GPUs")
                model = nn.DataParallel(model)
            else:
                logging.info("Force use one GPU")

        model.to(self.device)
        self.model = model
        self.model_ema = ModelEma(model, 0.99)
        self.inference_model = self.model_ema.ema

        self.num_class = 1

    def _run_prediction(self, dataset: Path) -> dict:
        with dataset.open("rb") as f:
            data = list(f)
        
        if 'mut_info' in json.loads(data[0]).keys(): 
            env_info = load_embedded_graph(data)
            
            label = env_info['label']
            pdb_codes = env_info['pdb_codes']
            mut_infos, chain_ids = zip(*[info.split("_", 1) for info in env_info['mut_infos']])
            feats = torch.from_numpy(np.array(env_info['feats'])).float().to(self.device)
            coords = torch.from_numpy(np.array(env_info['coords'])).float().to(self.device)
            mask = torch.from_numpy(np.array(env_info['mask'])).float().to(self.device)
            cas = torch.from_numpy(np.array(env_info['cas'])).float().to(self.device)
            from_aas = torch.from_numpy(np.array(env_info['from_aas'])).long().to(self.device)
            to_aas = torch.from_numpy(np.array(env_info['to_aas'])).long().to(self.device)
            input_aa = torch.concat(
                    (from_aas.reshape(-1, 1), to_aas.reshape(-1, 1)), dim=-1 )

            pp = np.array( [None] * mask.shape[0] )
            atom_types = np.array( [None] * mask.shape[0] )
        else:
            self.dms = True
            env_info = load_raw_graph(data)
            pdb_codes = env_info['pdb_codes']
            mut_infos, chain_ids = zip(*[info.split("_", 1) for info in env_info['mut_infos']])
            
            atom_types = env_info['atom_types'].to(self.device)
            coords = env_info['coords'].to(self.device)
            mask = env_info['mask'].to(self.device)
            cas = env_info['cas'].to(self.device)
            pp = env_info['pp'].to(self.device)
            input_aa = env_info['input_aa'].to(self.device)
            label = env_info['label']
            feats =  np.array( [None] * mask.shape[0] ) # no feats

        model_preds = []

        self.model.eval()
        with torch.no_grad():
            for idx in range(feats.shape[0] // self.args.batch_size + 1):
                bidx = self.args.batch_size * idx
                eidx = self.args.batch_size * idx + self.args.batch_size
                if eidx > feats.shape[0]:
                    eidx = feats.shape[0]
                pred, _ = self.model(
                    atom_types = atom_types[bidx:eidx],
                    pp = pp[bidx:eidx],
                    feats=feats[bidx:eidx],
                    ca=cas[bidx:eidx],
                    coords=coords[bidx:eidx],
                    mask=mask[bidx:eidx],
                    aa_feats=input_aa[bidx:eidx],
                )
                model_preds += pred.cpu().detach().numpy().tolist()
        
        df = pd.DataFrame(
            {
                "pdb_code": pdb_codes,
                "chain_id": chain_ids,
                "mutation": mut_infos,
                "exp_ddG": np.round(-np.array(label), 4),
                "pred_ddG": np.round(-np.array(model_preds), 4),
            }
        )

        return df
        

    def evaluate(self):
        df = self._run_prediction(self.args.dataset)

        label = df['exp_ddG'].values
        pred = df['pred_ddG'].values
        if not np.isnan(label[0]):
            logging.info(f"pearson corr: {round(pearsonr(pred, label)[0], 4)}")
            logging.info(f"spearman corr: {round(spearmanr(pred, label)[0], 4)}")
            logging.info(
            f"AUROC: {round(roc_auc_score(np.array(label) > 0.0, 1 / (1 + np.exp(-np.array(pred)))),4)}"
            )
        else:
            logging.info('No experimental ∆∆G labels provided to evaluate pearson, spearman, and AUROC')
        
        outfile_flat = self.args.outdir / f"{self.args.dataset.stem}_flat.csv"
        df.to_csv(outfile_flat, index=False)
        logging.info(f"Predictions wrote: {outfile_flat.resolve()}")

        if self.dms:
            outfile = self.args.outdir / f"{self.args.dataset.stem}.csv"
            format_dms_predictions(df, outfile)



    def inference(self, dataset: Path=None, outfile: Path= None) -> pd.DataFrame:

        if isinstance(dataset, Path):
            assert dataset.is_file(), f"Dataset file {dataset} not found"
            assert dataset.suffix == ".jsonl", f"Dataset file {dataset.name} must be a .jsonl file"
        else:
            dataset = self.args.dataset

        if not isinstance(outfile, Path):
            outfile = self.args.outdir / f"{dataset.stem}.csv"
            outfile_flat = self.args.outdir / f"{dataset.stem}_flat.csv"

        assert outfile.suffix == ".csv", f"Output file {outfile} must be a .csv file"

        df = self._run_prediction(dataset)
        df.to_csv(outfile_flat, index=False)

        logging.info(f"Predictions wrote to {outfile_flat}")
        
        if self.dms:
            format_dms_predictions(df, outfile)
            


 