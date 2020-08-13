#!/bin/bash

#SBATCH --job-name=seq2seq-transformer
#SBATCH --gres=gpu:4
#SBATCH --time=1-12:00
#SBATCH --mem=18G

# Declare the variables
# To run: sbatch --export=ALL seq2seq/run-seq2seq.sbatch

srun fairseq-train \
$bindatadir \
--arch transformer --share-decoder-input-output-embed \
--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
--lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
--dropout 0.3 --weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--max-tokens 7500 \
--max-epoch 40 \
--patience 10 \
--fp16 \
--tensorboard-logdir $modeldir/tblogs \
--save-dir $modeldir &> $log