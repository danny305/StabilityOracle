from typing import List
from dataclasses import dataclass, asdict
import itertools
import logging

import pandas as pd


@dataclass
class DataColumns:
    protein_col: str = "pdb_code"
    chain_col: str = "chain_id"
    pos_col: str = "position"
    wtAA_col: str = "wtAA"
    mutAA_col: str = "mutAA"
    from_col: str = "from"
    to_col: str = "to"


# Works for any phenotype where mutations at the same residue can be linearly combined
class MutationPermutationPipeline:
    def __init__(
        self,
        exp_columns: DataColumns,
        phenotypes: List[str] = None,
        permute: bool = True,
        **kwargs,
    ):
        assert isinstance(exp_columns, DataColumns), type(exp_columns)
        self.exp_cols = exp_columns

        self.protein_col = kwargs.get("protein_col", self.exp_cols.protein_col)
        self.chain_col = kwargs.get("chain_col", self.exp_cols.chain_col)
        self.pos_col = kwargs.get("position", self.exp_cols.pos_col)
        self.wt_col = kwargs.get("wt_col", self.exp_cols.wtAA_col)
        self.mut_col = kwargs.get("mut_col", self.exp_cols.mutAA_col)
        self.from_col = kwargs.get("from_col", self.exp_cols.from_col)
        self.to_col = kwargs.get("to_col", self.exp_cols.to_col)

        assert isinstance(phenotypes, (list, tuple, set))
        self.phenotypes = phenotypes

        self.permute = permute
        if self.permute:
            self.aug_name = "mut_perm"
        else:
            self.aug_name = "mut_comb"

    def _augment_phenotypes(self, verbose: bool = False) -> pd.DataFrame:
        self.temp_df[self.aug_name] = 0

        groups = self.temp_df.drop_duplicates(
            [
                self.protein_col,
                self.chain_col,
                self.pos_col,
                self.wt_col,
                self.mut_col,
            ]
        ).groupby(
            [
                self.protein_col,
                self.chain_col,
                self.pos_col,
                self.wt_col,
            ]
        )

        groups = [(name, df) for name, df in groups if len(df) > 1]

        if not "pymp" in dir():
            import pymp

        if self.permute:
            func = itertools.permutations
        else:
            func = itertools.combinations

        augmented_list = pymp.shared.list()
        with pymp.Parallel() as p:
            for idx in p.range(len(groups)):
                (pdb_id, chain_id, wtAA, position), mutant_df = groups[idx]

                mutants = list(row for _, row in mutant_df.iterrows())
                counts = 0
                for mut1, mut2 in func(mutants, 2):
                    if mut1[self.to_col] == mut2[self.to_col]:
                        # TODO print/log something here bc this should never happen
                        continue

                    if mut1[self.from_col] != mut2[self.from_col]:
                        # TODO print/log something here bc this should never happen
                        continue

                    mut = mut1.copy()
                    mut[self.from_col] = mut1[self.to_col]
                    mut[self.to_col] = mut2[self.to_col]

                    for phenotype in self.phenotypes:
                        mut[phenotype] = round(mut2[phenotype] - mut1[phenotype], 4)

                    mut[self.aug_name] = 1
                    augmented_list.append(mut)
                    counts += 1

                if verbose:
                    p.print(
                        f"Thread: {p.thread_num} Finished {counts}({len(mutant_df)}) Augment on: {pdb_id} {chain_id} {wtAA} {position}"
                    )

        augmented_list = list(augmented_list)
        self.augment_df = None
        if augmented_list:
            self.augment_df = pd.concat(augmented_list, axis=1).T
            logging.info(
                f"Generated {len(self.augment_df)} {self.aug_name} augmented mutations"
            )

    def augment_dataset(
        self, df: pd.DataFrame, return_concat: bool = False, **kwargs
    ) -> pd.DataFrame:
        for attr, col_name in asdict(self.exp_cols).items():
            assert col_name in df.columns, df.columns

        self.temp_df = df.copy()
        self.augment_df = None

        verbose = kwargs.get("verbose", False)

        logging.info(f"Generating augment mutations: {self.aug_name}")
        self._augment_phenotypes(verbose)

        self.concat_df = pd.concat([self.temp_df, self.augment_df])

        if return_concat:
            self.concat_df

        return self.augment_df


class ThermodynamicPermutationPipeline(MutationPermutationPipeline):

    def __init__(
        self,
        exp_columns: DataColumns,
        ddg_col_name: str = "ddG",
        permute: bool = True,
        **kwargs,
    ):
        assert isinstance(ddg_col_name, str), type(ddg_col_name)

        super().__init__(exp_columns, [ddg_col_name], permute, **kwargs)

        if self.permute:
            self.aug_name = "thermo_perm"
        else:
            self.aug_name = "thermo_comb"
