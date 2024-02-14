#!/bin/bash

data_dir=../data/datasets
out_dir=$data_dir/tp
for dataset in $data_dir/*.csv; 
do
    python run_permutation.py \
        --dataset $dataset \
        --outdir $out_dir \
        --n-threads 24
done