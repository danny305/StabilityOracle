from pathlib import Path
import argparse
import logging
import os

import pandas as pd

from StabilityOracle.augment import DataColumns, ThermodynamicPermutationPipeline


logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def cli():
    parser = argparse.ArgumentParser("Script to run Thermodynamic Permutations")

    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--outdir", default=Path.cwd() / "permutations", type=Path)
    parser.add_argument("--outfile", default=None, type=Path)

    parser.add_argument("--pdb-col", default="pdb_code", type=str)
    parser.add_argument("--chain-col", default="chain_id", type=str)
    parser.add_argument("--pos-col", default="position", type=str)
    parser.add_argument("--wt-col", default="wtAA", type=str)
    parser.add_argument("--mut-col", default="mutAA", type=str)
    parser.add_argument("--from-col", default="from", type=str)
    parser.add_argument("--to-col", default="to", type=str)
    parser.add_argument("--ddg-col", default="ddG", type=str)

    parser.add_argument("-n", "--n-threads", default=4, type=int)

    args = parser.parse_args()

    assert args.dataset.is_file(), args.dataset
    assert args.dataset.suffix == ".csv", args.outfile

    args.outdir.mkdir(0o770, parents=True, exist_ok=True)

    if args.outfile is None:
        args.outfile = args.outdir / f"{args.dataset.stem}_tp.csv"

    assert args.outfile.suffix == ".csv", args.outfile

    return args


if __name__ == "__main__":
    args = cli()
    logging.info(args)

    os.environ["PYMP_NUM_THREADS"] = str(args.n_threads)

    logging.info(f"Running Thermodynamic Permutations on {args.dataset.name}")

    data_cols = DataColumns(
        args.pdb_col,
        args.chain_col,
        args.pos_col,
        args.wt_col,
        args.mut_col,
        args.from_col,
        args.to_col,
    )

    pl = ThermodynamicPermutationPipeline(data_cols, args.ddg_col)

    df = pd.read_csv(args.dataset)

    aug_df = pl.augment_dataset(df)

    aug_df.to_csv(args.outfile, index=False)

    logging.info(f"Wrote augment dataset: {args.outfile}")
