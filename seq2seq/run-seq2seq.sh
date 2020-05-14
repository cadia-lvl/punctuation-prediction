#!/bin/bash -e

# Run a sequence 2 sequence transformer that learns to insert punctuation
cd seq2seq

input=rmh
casing=
ext=
if [ -n $casing ]; then
    ext=_$casing
fi
do_wer_tests=false

if [ "$input" = "rmh" ]; then
    datadir=/work/inga/data/${input}_subset/fairseq/sample55
    elif [ "$input" = "ep" ]; then
    datadir=/work/inga/data/europarl/processed/fairseq
else
    echo "Unrecognized input."
fi
export bindatadir=/work/inga/data/data-bin/${input}$ext.tokenized.nopuncts-${casing}puncts
export modeldir=/work/inga/punctuation/fairseq-out/${input}$ext/checkpoints
export log=$modeldir/log/train-$(date +'%Y%m%d').log

mkdir -p $modeldir $log

conda activate ptenv

# Prepare the data. casing is an optional argument
bash prepare-data-fairseqNMT.sh $datadir $input $casing

# Preprocess/binarize the data
TEXT=$datadir/${input}${casing}.tokenized
srun --mem=4G fairseq-preprocess --source-lang nopuncts --target-lang ${casing}puncts \
--trainpref $TEXT/train --validpref $TEXT/dev --testpref $TEXT/test \
--destdir $bindatadir --fp16 \
--workers 20 &

# Train a Transformer translation model
# Note: --max-tokens specifies the batch size
sbatch --export=log=$log,datadir=$datadir,modeldir=$modeldir run-seq2seq.sbatch

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

python ../utils/error_calculator.py \
$modeldir/target_punct2_style.txt $modeldir/test_predictions_punct2_style.txt \
> $modeldir/test_error.txt

if [ $do_wer_tests = "true" ]; then
    echo 'Apply the model on text with different WER inserted'
    bash wer-test-seq2seq.sh $datadir/wer_test$ext $modeldir/wer_test $input $casing
fi

exit 0;