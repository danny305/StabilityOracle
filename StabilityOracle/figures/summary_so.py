import numpy as np


colors = {
    "our": "#CC0000",
    "porstata": "#21618C",
    "thermonet": "#CCCC00",
    "prosgnn": "#00CCCC",
    "ddgun": "#CC6600",
    "premps": "#00CC00",
    "rosetta": "#CCCC00",
    "dynamut": "#2cb276",
    "sdm": "#d5558e",
    "foldx": "#0866b3",
    # "strum": "#E46C0A",
    "duet": "#5f5f5f",
}

t2837_ligand = {
    "pearson": 0.21,
    "spearman": 0.34,
    "rmse": 2.25,
    "accuracy": 0.65,
    "recall": 0.40,
    "precision": 0.39,
    "auc": 0.68,
}
t2837_nucleotides = {
    "pearson": 0.25,
    "spearman": 0.35,
    "rmse": 1.04,
    "accuracy": 0.78,
    "recall": 0.38,
    "precision": 0.58,
    "auc": 0.68,
}
t2837_protein = {
    "pearson": 0.28,
    "spearman": 0.36,
    "rmse": 2.09,
    "accuracy": 0.76,
    "recall": 0.52,
    "precision": 0.34,
    "auc": 0.71,
}


t2837_af = {
    "pearson": 0.58,
    "spearman": 0.62,
    "rmse": 1.66,
    "accuracy": 0.82,
    "recall": 0.55,
    "precision": 0.46,
    "auc": 0.81,
}
t2837_reverse_af = {
    "pearson": 0.58,
    "spearman": 0.62,
    "rmse": 1.65,
    "accuracy": 0.82,
    "recall": 0.86,
    "precision": 0.81,
    "auc": 0.81,
}
t2837aug_af = {
    "pearson": 0.66,
    "spearman": 0.69,
    "rmse": 1.52,
    "accuracy": 0.76,
    "recall": 0.70,
    "precision": 0.69,
    "auc": 0.84,
}
t2837aug_reverse_af = {
    "pearson": 0.66,
    "spearman": 0.69,
    "rmse": 1.54,
    "accuracy": 0.76,
    "recall": 0.8,
    "precision": 0.84,
    "auc": 0.84,
}


t2837 = {
    "pearson": 0.59,
    "spearman": 0.62,
    "rmse": 1.65,
    "accuracy": 0.82,
    "recall": 0.55,
    "precision": 0.46,
    "auc": 0.81,
}
t2837_reverse = {
    "pearson": 0.59,
    "spearman": 0.62,
    "rmse": 1.65,
    "accuracy": 0.82,
    "recall": 0.86,
    "precision": 0.82,
    "auc": 0.81,
}
t2837aug = {
    "pearson": 0.69,
    "spearman": 0.69,
    "rmse": 1.50,
    "accuracy": 0.77,
    "recall": 0.70,
    "precision": 0.69,
    "auc": 0.83,
}
t2837aug_reverse = {
    "pearson": 0.69,
    "spearman": 0.69,
    "rmse": 1.51,
    "accuracy": 0.76,
    "recall": 0.8,
    "precision": 0.84,
    "auc": 0.83,
}
ssym = {
    "pearson": 0.68,
    "spearman": 0.68,
    "rmse": 1.22,
    "accuracy": 0.83,
    "recall": 0.69,
    "precision": 0.52,
    "auc": 0.87,
}
ssym_reverse = {
    "pearson": 0.68,
    "spearman": 0.68,
    "rmse": 1.19,
    "accuracy": 0.80,
    "recall": 0.82,
    "precision": 0.95,
    "auc": 0.87,
}
ssymaug = {
    "pearson": 0.57,
    "spearman": 0.55,
    "rmse": 1.36,
    "accuracy": 0.71,
    "recall": 0.56,
    "precision": 0.78,
    "auc": 0.77,
}
ssymaug_reverse = {
    "pearson": 0.57,
    "spearman": 0.55,
    "rmse": 1.34,
    "accuracy": 0.72,
    "recall": 0.78,
    "precision": 0.78,
    "auc": 0.77,
}
myoglobin = {
    "pearson": 0.62,
    "spearman": 0.62,
    "rmse": 0.89,
    "accuracy": 0.71,
    "recall": 0.56,
    "precision": 0.36,
    "auc": 0.81,
}
myoglobin_reverse = {
    "pearson": 0.64,
    "spearman": 0.64,
    "rmse": 0.89,
    "accuracy": 0.72,
    "recall": 0.77,
    "precision": 0.94,
    "auc": 0.81,
}
fp = {
    "pearson": 0.58,
    "spearman": 0.62,
    "rmse": 1.48,
    "accuracy": 0.77,
    "recall": 0.61,
    "precision": 0.41,
    "auc": 0.79,
}
fp_reverse = {
    "pearson": 0.58,
    "spearman": 0.62,
    "rmse": 1.46,
    "accuracy": 0.76,
    "recall": 0.78,
    "precision": 0.94,
    "auc": 0.79,
}
fpaug = {
    "pearson": 0.66,
    "spearman": 0.64,
    "rmse": 1.32,
    "accuracy": 0.72,
    "recall": 0.65,
    "precision": 0.69,
    "auc": 0.80,
}
fpaug_reverse = {
    "pearson": 0.66,
    "spearman": 0.64,
    "rmse": 1.31,
    "accuracy": 0.72,
    "recall": 0.73,
    "precision": 0.80,
    "auc": 0.80,
}
s669 = {
    "pearson": 0.52,
    "spearman": 0.52,
    "rmse": 1.42,
    "accuracy": 0.76,
    "recall": 0.52,
    "precision": 0.38,
    "auc": 0.74,
}
s669_reverse = {
    "pearson": 0.52,
    "spearman": 0.52,
    "rmse": 1.41,
    "accuracy": 0.77,
    "recall": 0.80,
    "precision": 0.93,
    "auc": 0.74,
}
s669aug = {
    "pearson": 0.58,
    "spearman": 0.58,
    "rmse": 1.38,
    "accuracy": 0.73,
    "recall": 0.56,
    "precision": 0.42,
    "auc": 0.76,
}
s669aug_reverse = {
    "pearson": 0.58,
    "spearman": 0.58,
    "rmse": 1.37,
    "accuracy": 0.81,
    "recall": 0.59,
    "precision": 1.00,
    "auc": 0.76,
}
p53 = {
    "pearson": 0.73,
    "spearman": 0.62,
    "rmse": 1.50,
    "accuracy": 0.67,
    "recall": 0.45,
    "precision": 0.91,
    "auc": 0.77,
}
p53_reverse = {
    "pearson": 0.73,
    "spearman": 0.6,
    "rmse": 1.48,
    "accuracy": 0.74,
    "recall": 0.75,
    "precision": 0.95,
    "auc": 0.77,
}

c28aug_t2837 = {
    "pearson": 0.53,
    "spearman": 0.52,
    "rmse": 1.69,
    "accuracy": 0.78,
    "recall": 0.46,
    "precision": 0.44,
    "auc": 0.74,
}
c28aug_t2837_reverse = {
    "pearson": 0.52,
    "spearman": 0.52,
    "rmse": 1.67,
    "accuracy": 0.78,
    "recall": 0.74,
    "precision": 0.75,
    "auc": 0.74,
}
c28aug_t2837aug = {
    "pearson": 0.61,
    "spearman": 0.58,
    "rmse": 1.65,
    "accuracy": 0.74,
    "recall": 0.65,
    "precision": 0.70,
    "auc": 0.76,
}
c28aug_t2837aug_reverse = {
    "pearson": 0.60,
    "spearman": 0.58,
    "rmse": 1.63,
    "accuracy": 0.74,
    "recall": 0.77,
    "precision": 0.78,
    "auc": 0.76,
}

c28orig_t2837 = {
    "pearson": 0.51,
    "spearman": 0.50,
    "rmse": 1.71,
    "accuracy": 0.78,
    "recall": 0.43,
    "precision": 0.41,
    "auc": 0.72,
}
c28orig_t2837_reverse = {
    "pearson": 0.51,
    "spearman": 0.51,
    "rmse": 1.70,
    "accuracy": 0.78,
    "recall": 0.70,
    "precision": 0.82,
    "auc": 0.72,
}
c28orig_t2837aug = {
    "pearson": 0.58,
    "spearman": 0.55,
    "rmse": 1.68,
    "accuracy": 0.74,
    "recall": 0.62,
    "precision": 0.66,
    "auc": 0.74,
}
c28orig_t2837aug_reverse = {
    "pearson": 0.58,
    "spearman": 0.55,
    "rmse": 1.67,
    "accuracy": 0.74,
    "recall": 0.75,
    "precision": 0.76,
    "auc": 0.74,
}

c28scratch_t2837 = {
    "pearson": 0.34,
    "spearman": 0.31,
    "rmse": 2.08,
    "accuracy": 0.63,
    "recall": 0.26,
    "precision": 0.43,
    "auc": 0.64,
}
c28scratch_t2837_reverse = {
    "pearson": 0.34,
    "spearman": 0.31,
    "rmse": 2.13,
    "accuracy": 0.64,
    "recall": 0.85,
    "precision": 0.60,
    "auc": 0.64,
}
c28scratch_t2837aug = {
    "pearson": 0.45,
    "spearman": 0.40,
    "rmse": 1.86,
    "accuracy": 0.63,
    "recall": 0.50,
    "precision": 0.65,
    "auc": 0.69,
}
c28scratch_t2837aug_reverse = {
    "pearson": 0.45,
    "spearman": 0.40,
    "rmse": 1.90,
    "accuracy": 0.64,
    "recall": 0.76,
    "precision": 0.52,
    "auc": 0.69,
}

c28origscratch_t2837 = {
    "pearson": 0.30,
    "spearman": 0.30,
    "rmse": 2.25,
    "accuracy": 0.55,
    "recall": 0.26,
    "precision": 0.43,
    "auc": 0.63,
}
c28origscratch_t2837_reverse = {
    "pearson": 0.30,
    "spearman": 0.20,
    "rmse": 2.11,
    "accuracy": 0.66,
    "recall": 0.82,
    "precision": 0.62,
    "auc": 0.63,
}
c28origscratch_t2837aug = {
    "pearson": 0.4,
    "spearman": 0.37,
    "rmse": 1.98,
    "accuracy": 0.57,
    "recall": 0.45,
    "precision": 0.65,
    "auc": 0.67,
}
c28origscratch_t2837aug_reverse = {
    "pearson": 0.4,
    "spearman": 0.36,
    "rmse": 1.91,
    "accuracy": 0.64,
    "recall": 0.73,
    "precision": 0.45,
    "auc": 0.67,
}


pro_q1744_ssym = {
    "pearson": 0.51,
    "spearman": 0.47,
    "rmse": 1.40,
    "accuracy": 0.72,
    "recall": 0.43,
    "precision": 0.10,
    "auc": 0.71,
}
pro_q1744_ssym_reverse = {
    "pearson": 0.51,
    "spearman": 0.46,
    "rmse": 1.40,
    "accuracy": 0.72,
    "recall": 0.84,
    "precision": 0.62,
    "auc": 0.72,
}
pro_q1744_myoglobin = {
    "pearson": 0.48,
    "spearman": 0.45,
    "rmse": 1.01,
    "accuracy": 0.68,
    "recall": 0.19,
    "precision": 0.05,
    "auc": 0.70,
}
pro_q1744_myoglobin_reverse = {
    "pearson": 0.47,
    "spearman": 0.45,
    "rmse": 1.01,
    "accuracy": 0.69,
    "recall": 0.89,
    "precision": 0.63,
    "auc": 0.70,
}
pro_q1744_p53 = {
    "pearson": 0.59,
    "spearman": 0.59,
    "rmse": 1.77,
    "accuracy": 0.71,
    "recall": 0.64,
    "precision": 0.17,
    "auc": 0.71,
}
pro_q1744_p53_reverse = {
    "pearson": 0.59,
    "spearman": 0.59,
    "rmse": 1.77,
    "accuracy": 0.78,
    "recall": 0.74,
    "precision": 0.55,
    "auc": 0.78,
}
pro_q1744_s669 = {
    "pearson": 0.49,
    "spearman": 0.50,
    "rmse": 1.45,
    "accuracy": 0.75,
    "recall": 0.37,
    "precision": 0.5,
    "auc": 0.73,
}
pro_q1744_t2837 = {
    "pearson": 0.42,
    "spearman": 0.45,
    "rmse": 1.52,
    "accuracy": 0.73,
    "recall": 0.43,
    "precision": 0.36,
    "auc": 0.72,
}
pro_q1744_t2837aug = {
    "pearson": 0.52,
    "spearman": 0.55,
    "rmse": 1.93,
    "accuracy": 0.69,
    "recall": 0.67,
    "precision": 0.57,
    "auc": 0.78,
}


pro_t2837 = {
    "pearson": 0.53,
    "spearman": 0.53,
    "rmse": 1.75,
    "accuracy": 0.77,
    "recall": 0.44,
    "precision": 0.46,
    "auc": 0.75,
}
pro_t2837aug = {
    "pearson": 0.67,
    "spearman": 0.65,
    "rmse": 1.49,
    "accuracy": 0.75,
    "recall": 0.67,
    "precision": 0.69,
    "auc": 0.81,
}

onehot_t2837 = {
    "pearson": 0.48,
    "spearman": 0.44,
    "rmse": 1.90,
    "accuracy": 0.66,
    "recall": 0.35,
    "precision": 0.44,
    "auc": 0.68,
}
onehot_t2837_aug = {
    "pearson": 0.53,
    "spearman": 0.49,
    "rmse": 1.83,
    "accuracy": 0.62,
    "recall": 0.43,
    "precision": 0.48,
    "auc": 0.71,
}
muteverything_t2837 = {
    "pearson": 0.63,
    "spearman": 0.65,
    "rmse": 1.58,
    "accuracy": 0.79,
    "recall": 0.56,
    "precision": 0.48,
    "auc": 0.80,
}

rasp_t2837 = {
    "pearson": 0.58,
    "spearman": 0.54,
    "rmse": 1.55,
    "accuracy": 0.79,
    "recall": 0.44,
    "precision": 0.33,
    "auc": 0.61,
}
rasp_t2837aug = {
    "pearson": 0.49,
    "spearman": 0.46,
    "rmse": 1.88,
    "accuracy": 0.66,
    "recall": 0.69,
    "precision": 0.38,
    "auc": 0.66,
}


score_af_dict = {
    "t2837": t2837_af,
    "t2837_reverse": t2837_reverse_af,
    "t2837aug": t2837aug_af,
    "t2837aug_reverse": t2837aug_reverse_af,
}
c28origscratch_dict = {
    "t2837": c28origscratch_t2837,
    "t2837_reverse": c28origscratch_t2837_reverse,
    "t2837aug": c28origscratch_t2837aug,
    "t2837aug_reverse": c28origscratch_t2837aug_reverse,
}
c28scratch_dict = {
    "t2837": c28scratch_t2837,
    "t2837_reverse": c28scratch_t2837_reverse,
    "t2837aug": c28scratch_t2837aug,
    "t2837aug_reverse": c28scratch_t2837aug_reverse,
}
c28orig_dict = {
    "t2837": c28orig_t2837,
    "t2837_reverse": c28orig_t2837_reverse,
    "t2837aug": c28orig_t2837aug,
    "t2837aug_reverse": c28orig_t2837aug_reverse,
}
c28aug_dict = {
    "t2837": c28aug_t2837,
    "t2837_reverse": c28aug_t2837_reverse,
    "t2837aug": c28aug_t2837aug,
    "t2837aug_reverse": c28aug_t2837aug_reverse,
}
prostata_dict = {"t2837": pro_t2837, "t2837aug": pro_t2837aug}
rasp_dict = {"t2837": rasp_t2837, "t2837aug": rasp_t2837aug}
onehot_dict = {"t2837": onehot_t2837, "t2837aug": onehot_t2837_aug}
muteverything_dict = {"t2837": muteverything_t2837}

score_dict = {
    "t2837": t2837,
    "t2837_reverse": t2837_reverse,
    "ssym": ssym,
    "ssym_reverse": ssym_reverse,
    "ssymaug": ssymaug,
    "ssymaug_reverse": ssymaug_reverse,
    "s669": s669,
    "s669_reverse": s669_reverse,
    "s669aug": s669aug,
    "s669aug_reverse": s669aug_reverse,
    "fp": fp,
    "fp_reverse": fp_reverse,
    "fpaug": fpaug,
    "fpaug_reverse": fpaug_reverse,
    "p53": p53,
    "p53_reverse": p53_reverse,
    "myoglobin": myoglobin,
    "myoglobin_reverse": myoglobin_reverse,
    "t2837aug": t2837aug,
    "t2837aug_reverse": t2837aug_reverse,
}


mark = {
    "our": "o",
    "porstata": "p",
    "thermonet": "<",
    "prosgnn": "*",
    "ddgun": "^",
    "premps": "o",
    "rosetta": ">",
    "dynamut": "d",
    "sdm": "s",
    "foldx": ">",
    "strum": "D",
    "duet": "d",
    "rasp": "^",
}


labels = {
    "rasp": "RaSP",
    "our": "SO",
    "porstata": "PROSTATA",
    "thermonet": "Thermonet",
    "prosgnn": "ProsGNN",
    "ddgun": "DDGun",
    "premps": "PremPS",
    "rosetta": "Rosetta",
    "dynamut": "Dynamut",
    "sdm": "SDM",
    "foldx": "FoldX",
    "strum": "Strum",
    "duet": "DUET",
}
