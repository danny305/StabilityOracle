#!/bin/bash

# Run all datasets
# requires downloading all graph datasets and placing them in ../data/graphs
outdir=../data/predictions
for dataset in ../data/graphs/*.jsonl; 
do 
    name=$(basename $dataset)
    outfile="$outdir/${name%.*}.csv"
    if [[ -e $outfile ]]; then
        echo "Prediction file exist: $outfile"
    else
        echo "Generating predictions for $dataset"

        python run_stability_oracle.py \
            --model-ckpt ../models/stability_oracle.pt \
            --batch-size 240 \
            --use-clstoken \
            --outdir $outdir \
            --dataset $dataset

    fi
done

# Run single dataset
# requires downloading graph datasets and placing it in ../data/graphs
# dataset=../data/graphs/p53.jsonl
# outdir=../data/predictions
# python run_stability_oracle.py \
#         --model-ckpt ../models/stability_oracle.pt \
#         --batch-size 240 \
#         --use-clstoken \
#         --outdir ../data/predictions \
#         --dataset $dataset
