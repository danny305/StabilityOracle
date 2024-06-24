from pathlib import Path
import logging

import pandas as pd
import numpy as np

from StabilityOracle.model.dataloader import AA_3to1

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

_reverse_amino_acids = {
    _amino_acids[key]: key
    for key in _amino_acids.keys()
}



def format_dms_predictions(df: pd.DataFrame, output_csv: Path):
    head = [
    "cif_file",
    "model_id",
    "chain_id",
    "res_seq_num",
    "wt_AA",
    "pred_AA",
    "pred_prob",
    "prALA",
    "prARG",
    "prASN",
    "prASP",
    "prCYS",
    "prGLN",
    "prGLU",
    "prGLY",
    "prHIS",
    "prILE",
    "prLEU",
    "prLYS",
    "prMET",
    "prPHE",
    "prPRO",
    "prSER",
    "prTHR",
    "prTRP",
    "prTYR",
    "prVAL",
    ]
    data_dict = {}
    for item in head:
        data_dict[item] = []
    
    AA_1to3 = {}
    for key in AA_3to1.keys():
        AA_1to3[ AA_3to1[key] ] = key

    loop_idx = len(_amino_acids.keys()) # number of aas
    for idx, row in df.iterrows():
        if idx % loop_idx == 0:
            wt = row['mutation'][0]
            preds = []

            data_dict["cif_file"].append( row['pdb_code'] )
            data_dict["model_id"].append( np.nan )
            data_dict["chain_id"].append( row['chain_id'] ) 
            data_dict["res_seq_num"].append( int(row['mutation'][1:-1]) )
            data_dict["wt_AA"].append( AA_1to3[row['mutation'][0]] )
        preds.append( ( row['pred_ddG'], AA_1to3[row['mutation'][-1]] ) )
        
        if len( preds ) == loop_idx:
            data_dict["pred_AA"].append( sorted(preds, key = lambda x: x[0], reverse=True)[0][1] )
            data_dict["pred_prob"].append( sorted(preds, key = lambda x: x[0], reverse=True)[0][0] )
        
        to_type = AA_1to3[ row['mutation'][-1]  ]
        data_dict['pr'+to_type].append( row['pred_ddG'] )
    
    df = pd.DataFrame(data_dict)
    df.to_csv(output_csv, index=False, float_format="%.4f")

    logging.info(f"Predictions wrote: {output_csv.resolve()}")
