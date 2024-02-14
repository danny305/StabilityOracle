import numpy as np


colors = {
    "our": "#CC0000",
    "our_single": "#CC0000",
    "porstata": "#21618C",
    "thermonet": "#7e252f",  # "#CCCC00",
    "prosgnn": "#00CCCC",
    "ddgun": "#CC6600",
    "premps": "#00CC00",
    "rosetta": "#CCCC00",
    "dynamut": "#2cb276",
    "sdm": "#d5558e",
    "foldx": "#0866b3",
    "rasp": "#6c65d2",  # "#E46C0A",
    "duet": "#5f5f5f",
}


ssym = {
    "our": 0.72,
    "our_single": 0.68,
    "our_c28": 0.63,
    "porstata_c28": 0.51,
    "porstata": 0.56,
    "thermonet": 0.47,
    "prosgnn": 0.61,
    "ddgun": 0.48,
    "sdm": 0.51,
    "rosetta": 0.68,
    "duet": 0.63,
    "foldx": 0.63,
    "rasp": 0.64,
}
ssym_reverse = {
    "our": 0.72,
    "our_single": 0.68,
    "our_c28": 0.62,
    "porstata_c28": 0.51,
    "porstata": 0.56,
    "thermonet": 0.47,
    "prosgnn": 0.56,
    "ddgun": 0.48,
    "sdm": 0.32,
    "rosetta": 0.43,
    "duet": 0.13,
    "foldx": 0.39,
    "rasp": 0.30,
}

s669 = {
    "our": 0.52,
    "our_single": 0.52,
    "our_c28": 0.50,
    "porstata_c28": 0.48,
    "porstata": 0.51,
    "thermonet": 0.39,
    "ddgun": 0.41,
    "premps": 0.41,
    "rosetta": 0.39,
    "dynamut": 0.41,
    "sdm": 0.41,
    "duet": 0.41,
    "rasp": 0.39,
}

s669_reverse = {
    "our": 0.52,
    "our_single": 0.52,
    "our_c28": 0.50,
    "porstata_c28": 0.48,
    "porstata": 0.51,
    "thermonet": 0.38,
    "ddgun": 0.38,
    "premps": 0.42,
    "rosetta": 0.4,
    "dynamut": 0.34,
    "sdm": 0.13,
    "duet": 0.23,
    "rasp": 0.27,
}

p53 = {
    "our": 0.73,
    "our_single": 0.73,
    "our_c28": 0.64,
    "porstata_c28": 0.59,
    "porstata": 0.73,
    "prosgnn": 0.62,
    "thermonet": 0.45,
    "rasp": 0.67,
}

p53_reverse = {
    "our": 0.73,
    "our_single": 0.73,
    "our_c28": 0.64,
    "porstata_c28": 0.59,
    "porstata": 0.73,
    "prosgnn": 0.60,
    "thermonet": 0.56,
    "rasp": 0.10,
}
proteing = {"our": 0.75, "foldx": 0.51, "rosetta": 0.64, "rasp": 0.74, "porstata": 0.78}
proteing_reverse = {"our": 0.75, "porstata": 0.78, "rasp": 0.29}

myoglobin = {
    "our": 0.68,
    "our_single": 0.62,
    "our_c28": 0.55,
    "porstata_c28": 0.48,
    "porstata": 0.55,
    "prosgnn": 0.48,
    "thermonet": 0.38,
    "rasp": 0.68,
}
myoglobin_reverse = {
    "our": 0.68,
    "our_single": 0.64,
    "our_c28": 0.55,
    "porstata_c28": 0.48,
    "porstata": 0.55,
    "prosgnn": 0.43,
    "thermonet": 0.37,
    "rasp": 0.36,
}
t2837 = {
    "our": 0.59,
    "our_single": 0.58,
    "porstata": 0.53,
    "our_c28": 0.53,
    "porstata_c28": 0.44,
    "rasp": 0.55,
}
t2837_reverse = {
    "our": 0.59,
    "our_single": 0.58,
    "porstata": 0.53,
    "our_c28": 0.52,
    "porstata_c28": 0.44,
    "rasp": 0.23,
}
s669aug = {
    "our": 0.58,
    "our_single": 0.58,
    "porstata": 0.58,
    "our_c28": 0.53,
    "porstata_c28": 0.51,
}
s669aug_reverse = {
    "our": 0.58,
    "our_single": 0.58,
    "porstata": 0.58,
    "our_c28": 0.53,
    "porstata_c28": 0.51,
}
fp = {
    "our": 0.61,
    "our_single": 0.59,
    "porstata": 0.52,
    "our_c28": 0.53,
    "porstata_c28": 0.50,
}
fp_reverse = {
    "our": 0.61,
    "our_single": 0.59,
    "porstata": 0.52,
    "our_c28": 0.53,
    "porstata_c28": 0.50,
}
fpaug = {
    "our": 0.69,
    "our_single": 0.65,
    "porstata": 0.60,
    "our_c28": 0.58,
    "porstata_c28": 0.54,
}
fpaug_reverse = {
    "our": 0.69,
    "our_single": 0.65,
    "porstata": 0.60,
    "our_c28": 0.59,
    "porstata_c28": 0.54,
}

score_dict = {
    "t2837": t2837,
    "t2837_reverse": t2837_reverse,
    "ssym": ssym,
    "ssym_reverse": ssym_reverse,
    "s669": s669,
    "s669_reverse": s669_reverse,
    "p53": p53,
    "p53_reverse": p53_reverse,
    "myoglobin": myoglobin,
    "myoglobin_reverse": myoglobin_reverse,
}

# remove single our
for dict_key in score_dict:
    if "our_single" in score_dict[dict_key].keys():
        del score_dict[dict_key]["our_single"]
        del score_dict[dict_key]["our_c28"]
        del score_dict[dict_key]["porstata_c28"]

mark = {
    "our": "o",
    "porstata": "p",
    "thermonet": "*",
    "prosgnn": "*",
    "ddgun": "^",
    "premps": "o",
    "rosetta": ">",
    "dynamut": "d",
    "sdm": "s",
    "foldx": ">",
    "rasp": "D",
    "duet": "d",
}


labels = {
    "our": "Stability-Oracle",
    "porstata": "PROSTATA-IFML",
    "thermonet": "Thermonet",
    "prosgnn": "ProsGNN",
    "ddgun": "DDGun",
    "premps": "PremPS",
    "rosetta": "Rosetta",
    "dynamut": "Dynamut",
    "sdm": "SDM",
    "foldx": "FoldX",
    "rasp": "RaSP",
    "duet": "DUET",
}
