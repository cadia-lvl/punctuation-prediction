#!/bin/bash -e

# Run a fairseq sequence-to-sequence transformer that learns to insert punctuation.
# Following: https://github.com/pytorch/fairseq/blob/master/examples/translation/README.md#training-a-new-model
# The Icelandic Gigaword corpus data is obtained with ../process/rmh_subset_specific.ipynb
# and cleaned with rmh_data_cleaning.sh
# The English Europarl data is obtained and cleaned with ../process/europarl_cleaning.sh

conda activate ptenv

# git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
cd ..

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

# Evaluate our trained model
srun --mem 8G --time 0-08:00 \
fairseq-generate $bindatadir \
--path $modeldir/checkpoint_best.pt \
--batch-size 64 --beam 5 --remove-bpe \
&>$modeldir/test.out &

# Extract the hypotheses and references and calculate an F-score
egrep "^D-" $modeldir/test.out | cut -f3- > $modeldir/test_predictions.txt
egrep "^T-" $modeldir/test.out | cut -f2- > $modeldir/target.txt

# Check if the model changes regular words and how many
wdiff -3 <(sed -r 's:PERIOD|COMMA|QUESTIONMARK::g' $modeldir/target.txt) \
<(sed -r 's:PERIOD|COMMA|QUESTIONMARK::g' $modeldir/test_predictions.txt) |egrep -v '=|^$' |wc -l

# Use the Punctuator2 F1-score calculator
sed -e 's:COMMA:,COMMA:g' \
-e 's:QUESTIONMARK:\?QUESTIONMARK:g' \
-e 's:PERIOD:.PERIOD:g'\
< $modeldir/target.txt \
| tr '\n' ' ' | sed -r 's: +: :g' \
> $modeldir/target_punct2_style.txt

sed -e 's:COMMA:,COMMA:g' \
-e 's:QUESTIONMARK:\?QUESTIONMARK:g' \
-e 's:PERIOD:.PERIOD:g'\
< $modeldir/test_predictions.txt \
| tr '\n' ' ' | sed -r 's: +: :g' \
> $modeldir/test_predictions_punct2_style.txt

python utils/error_calculator.py \
$modeldir/target_punct2_style.txt $modeldir/test_predictions_punct2_style.txt \
> $modeldir/test_error.txt

if [ $do_wer_tests = "true" ]; then
    echo 'Apply the model on text with different WER inserted'
    bash seq2seq/wer-test-seq2seq.sh $datadir/wer_test$ext $modeldir/wer_test $input $casing
fi

exit 0;