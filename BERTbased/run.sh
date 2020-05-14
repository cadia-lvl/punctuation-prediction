#!/bin/bash -e

set -o pipefail

# Run a hugging face ner transformer on a punctuation task

conda activate tf20env

git clone https://github.com/huggingface/transformers
cd transformers
pip install .
pip install seqeval
pip install git+https://github.com/fastai/fastprogress.git

cd -
input=rmh  #ep
do_wer_tests=false

if [ "$input" = "rmh" ]; then
    orig=/work/inga/data/rmh_subset/punctuator/sample55
    export DATA_DIR=/work/inga/data/rmh_subset/NERtrans
    elif [ "$input" = "ep" ]; then
    orig=/work/inga/data/europarl/processed
    export DATA_DIR=/work/inga/data/europarl/processed/NERtrans
else
    echo "Unrecognized input."
fi

error_calc_dir=utils
export SCRIPT_DIR=transformers/examples/ner
export MAX_LENGTH=60
export MAX_SEQ_LENGTH=180
export BERT_MODEL=bert-base-multilingual-cased
export OUTPUT_DIR=/work/inga/punctuation/NERtrans-out/$input-transformer-model
export BATCH_SIZE=16
export NUM_EPOCHS=3
export SAVE_STEPS=3000
export SEED=42

tmp=$DATA_DIR/tmp
mkdir -p $tmp
d=$(date +'%Y%m%d')

echo "Get the data on the right format"
# 1. Put one word (+ a possible punct) per line,
# 2. add a space between the punct token and the word,
# 3-7. Change the punct symbols from the punctuator 2 look and map puncts to periods, commas and question marks
# 8. Remove troublesome unicode symbols and weird symbols (should be changed to remove everything except accepted symbols)
# 9. Keep only the first two columns (there can be more than one punctuation after a word)
# 10. Remove lines containing only a label or empty
# 11. Add O to empty slots in the 2nd column
for i in train dev test; do
    (
        sed -r 's: ([\.,\?\!\:\;\-][A-Z]{4,}):\1:g' $orig/${input}.${i}.txt | tr ' ' '\n' \
        | sed -re 's:([\.,\?\!\:\;\-][A-Z]{4,}): \1:g' \
        -e 's:\-DASH|\:COLON:COMMA:g' \
        -e 's:\;SEMICOLON|\!EXCLAMATIONMARK:PERIOD:g' \
        -e 's:,COMMA:COMMA:g' \
        -e 's:\?QUESTIONMARK:QUESTIONMARK:g' \
        -e 's:\.PERIOD:PERIOD:g' \
        -e 's/\)xc2\xad|\xc2\x8d|\xc2\x90|\xc2\x93|\xc2\x9d|\x60|´|‟|‐|~|‑|‒|—|―|−|•|’|⋄|±|×||·|®|©||| //g' \
        | cut -d' ' -f1,2 \
        | egrep -v '^ |^$' \
        | awk -F' ' '{if ($2 == "") print $1,"O"; else print $0}' \
        > $tmp/$i.txt.tmp
    ) &
done
wait;

echo "Split the data into sequences of length $MAX_LENGTH."
for i in train dev test; do
    awk -v m=$MAX_LENGTH ' {print;} NR % m == 0 { print ""; }' $tmp/$i.txt.tmp > ${DATA_DIR}/$i.txt
done

echo "Create a labels file"
cat ${DATA_DIR}/{train.txt,dev.txt,test.txt} | cut -d " " -f 2 | egrep -v "^O?$"| sort | uniq > ${DATA_DIR}/labels.txt

echo "Train the tensorflow NER model"
# Get the following error when use more than one GPU:
# https://github.com/tensorflow/tensorflow/issues/35100
sbatch \
--job-name=${input}-NER-transformer \
--nodelist=torpaq --partition=longrunning \
--output=${DATA_DIR}/tf_${input}_transformer_$d.log \
--gres=gpu:1 --mem=28G --time=2-00:00 \
--wrap="srun \
python3 ${SCRIPT_DIR}/run_tf_ner.py \
--data_dir ${DATA_DIR}/ \
--model_type bert \
--labels ${DATA_DIR}/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_SEQ_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_device_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--fp16 \
--gpus '0' \
--do_train \
--do_eval \
--do_predict"

# Use the Punctuator2 F1-score calculator
sed -re 's: O$::' -e 's:COMMA:,COMMA:' \
-e 's:QUESTIONMARK:\?QUESTIONMARK:' \
-e 's:PERIOD:.PERIOD:'\
< $OUTPUT_DIR/test_predictions.txt \
| tr '\n' ' ' | sed -r 's: +: :g' \
> $OUTPUT_DIR/test_predictions_theano_style.txt

echo 'Calculate F1-scores'
python $error_calc_dir/error_calculator.py \
$orig/${input}.test.txt $OUTPUT_DIR/test_predictions_theano_style.txt \
> $OUTPUT_DIR/test_punct2error.txt

# NOTE! The following can't be run like this. Need to find another way
# if [ $do_wer_tests = "true" ]; then
#     echo 'Apply the model on text with different WER inserted'
#     bash wer-test.sh $input $DATA_DIR
# fi

exit 0;