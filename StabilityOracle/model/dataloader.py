import json
import torch
import pandas as pd
import numpy as np


ELEMENT_MAP = lambda x: {
    "H": 0,
    "C": 1,
    "N": 2,
    "O": 3,
    "F": 4,
    "S": 5,
    "P": 6,
    "F": 7,
    "Cl": 7,
    "CL": 7,
    "Br": 7,
    "BR": 7,
    "I": 7,
}.get(x, 8)

AA_INDEX = {
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
AA_3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

AMINO_ACIDS = list(AA_INDEX.keys()) 

def load_embedded_graph(jsonl: dict, device: torch.device=None) -> dict:
    feats = []
    coords = []
    mask = []
    cas = []
    label = []
    from_aas = []
    to_aas = []
    mut_infos = []
    pdb_codes = []
    
    
    for data_idx in range(len(jsonl)):
        example = json.loads(jsonl[data_idx])

        mut_infos.append(example["mut_info"])
        pdb_codes.append(example["pdb_id"])
        feats.append(example["input"])
        coords.append(example["coords"])
        mask.append(example["mask"])
        cas.append(example["ca"])
        label.append(-example["ddg"])
        from_aas.append(example["from"])
        to_aas.append(example["to"])

    return {
        "feats": feats,
        "coords": coords,
        "mask": mask,
        "cas": cas,
        "label": label,
        "from_aas": from_aas,
        "to_aas": to_aas,
        "mut_infos": mut_infos,
        "pdb_codes": pdb_codes,
    }


def load_raw_graph(jsonl: dict, device: torch.device=None) -> dict:
    max_atoms = 512

    PPs = []
    ATs = []
    COORDs = []
    masks = []
    CAs = []
    labels = []
    mutation_AAs = []
    mut_infos = []
    pdb_codes = []

    
    for data_idx in range(len(jsonl)):
        data = json.loads(jsonl[data_idx])
    
        ss = data.get("snapshot", {})
        filename = ss.get("filename", "")
        model_id = ss.get("model_id", 1)
    
        atoms = pd.DataFrame.from_dict(data["atomic_collection"])
        ca_idx = data.get("target_alpha_carbon", None)
        masked_aa_atom_idx = set(data["target"])
        

        wt_AA = ss.get("wt_aa", None)
        if wt_AA is None: wt_AA = ss.get("label", "")

        atom_index = {idx for idx in range(len(atoms))}
        atom_index = list(atom_index - masked_aa_atom_idx)

        ca = (
            torch.as_tensor(
                [
                    atoms["x"][ca_idx],
                    atoms["y"][ca_idx],
                    atoms["z"][ca_idx],
                ],
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .repeat(20, 1)
        )

        pos = atoms["res_seq_num"][ca_idx]
        chain_id = atoms["chain_id"][ca_idx]

        atoms = atoms.iloc[atom_index].reset_index(drop=True)
        if atoms.shape[0] > max_atoms:
            atoms = atoms[: max_atoms]

        coords = torch.as_tensor(
            atoms[["x", "y", "z"]].to_numpy(), dtype=torch.float32
        )

        coords = (
            torch.cat(
                (coords, torch.zeros([max_atoms - coords.shape[0], 3])),
                dim=0,
            )
            .float()
            .unsqueeze(0)
            .repeat(20, 1, 1)
        )

        atom_types = torch.as_tensor(
            list(map(ELEMENT_MAP, atoms.element)), dtype=torch.long
        )

        atom_types = (
            torch.cat(
                (
                    atom_types,
                    torch.zeros([max_atoms - atom_types.shape[0]]).long(),
                ),
                dim=0,
            )
            .unsqueeze(0)
            .repeat(20, 1)
        )
        
        add_props = ['_atom_site.fw2_charge', '_atom_site.FreeSASA_value'] 
        for prop in add_props:
            atoms.loc[atoms[prop].isin(['?', '.']), prop] = 0.0
            atoms[prop] = atoms[prop].astype(float)
        
        pp = torch.as_tensor(atoms[add_props].to_numpy(),dtype=torch.float32,)
        pp = (
            torch.cat((pp, torch.zeros([max_atoms - pp.shape[0], 2])), dim=0)
            .float()
            .unsqueeze(0)
            .repeat(20, 1, 1)
        )

        # Do not mask any atoms
        mask = torch.ones([max_atoms]).float().unsqueeze(0).repeat(20, 1)
        
        wt_AA = ss['label']
        
        from_to_pairs = torch.as_tensor(
            [ [ AA_INDEX[wt_AA], AA_INDEX[to_AA] ] for to_AA in AMINO_ACIDS],
            dtype=torch.long,
        )
        pdb_code = filename[:4]
        pos = ss.get('res_seq_num', 0) 
        chain_id = ss.get('chain_id', 'A')
        mut_info = [AA_3to1[wt_AA] + str(pos) + AA_3to1[to_AA] + '_' + chain_id for to_AA in AMINO_ACIDS]
        label = ss.get('ddg', np.nan) 

        ATs.append(atom_types)
        COORDs.append(coords)
        CAs.append(ca)
        PPs.append(pp)
        masks.append(mask)
        mutation_AAs.append( from_to_pairs )
        pdb_codes += [ pdb_code ] * 20
        mut_infos += mut_info 
        labels += [label] * 20
    
    
    return {
        'atom_types': torch.cat(ATs, dim=0),
        'coords': torch.cat(COORDs, dim=0),
        'cas': torch.cat(CAs, dim=0),
        'pp': torch.cat(PPs, dim=0),
        'mask': torch.cat(masks, dim=0),
        'input_aa': torch.cat(mutation_AAs, dim=0),
        'pdb_codes': pdb_codes, 
        'mut_infos': mut_infos,
        'label': labels
    }
