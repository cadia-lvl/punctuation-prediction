#!/bin/bash -e

# Run a fairseq sequence-to-sequence transformer that learns to insert punctuation.
# Following: https://github.com/pytorch/fairseq/blob/master/examples/translation/README.md#training-a-new-model
# The Icelandic Gigaword corpus data is obtained with ../process/rmh_subset_specific.ipynb
# and cleaned with rmh_data_cleaning.sh
# The English Europarl data is obtained and cleaned with ../process/europarl_cleaning.sh

input=rmh # Choose yourself
casing=
ext=
if [ -n $casing ]; then
    ext=_$casing
fi
do_wer_tests=false

datadir=./data/processed/${input}/seq2seq
export bindatadir=./data-bin/${input}$ext.tokenized.nopuncts-${casing}puncts
export modeldir=./out/seq2seq-out/${input}$ext/checkpoints
logdir=$modeldir/log
export log=$logdir/train-$(date +'%Y%m%d').log

mkdir -p $modeldir $logdir

# Prepare the data. casing is an optional argument
srun --mem=8G bash seq2seq/prepare-data-fairseqNMT.sh $datadir $input $casing &> $datadir/prepare-data.log

# Preprocess/binarize the data
TEXT=$datadir/${input}${casing}.tokenized
srun --mem=8G fairseq-preprocess --source-lang nopuncts --target-lang ${casing}puncts \
--trainpref $TEXT/train --validpref $TEXT/dev --testpref $TEXT/test \
--destdir $bindatadir --fp16 \
--workers 20

# Train a Transformer translation model
# Note: --max-tokens specifies the batch size
sbatch --export=ALL seq2seq/run-seq2seq.sbatch

echo 'After the training is finished generate predictions with generate.sh'
exit 0;