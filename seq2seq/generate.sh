#!/bin/bash -e

# Run a sequence 2 sequence transformer that learns to insert punctuation
input=rmh
casing=
if [ -n $casing ]; then
    ext=_$casing
fi
do_wer_tests=false

datadir=./data/processed/${input}/seq2seq
bindatadir=./data-bin/${input}$ext.tokenized.nopuncts-${casing}puncts
modeldir=./out/seq2seq-out/${input}$ext/checkpoints

# Evaluate our trained model
srun --mem 8G --time 0-12:00 \
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

python utils/error_calculator.py --transformer \
$modeldir/target.txt $modeldir/test_predictions.txt \
> $modeldir/test_error.txt

if [ $do_wer_tests = "true" ]; then
    echo 'Apply the model on text with different WER inserted'
    bash $scriptdir/seq2seq/wer-test-seq2seq.sh $datadir/wer_test$ext $modeldir/wer_test $input $casing
fi

exit 0;