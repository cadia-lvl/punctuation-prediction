#!/bin/bash -e

set -o pipefail

# Run a hugging face ner transformer on a punctuation task as explained here:
# https://github.com/huggingface/transformers/tree/master/examples/token-classification

# The English Europarl data is cleaned with ../process/europarl_cleaning.sh

# Activate the virtual environment
#conda activate ptenv #tf21env

#pip install seqeval
#pip install git+https://github.com/fastai/fastprogress.git
#git clone https://github.com/huggingface/transformers
#cd transformers
#pip install .
#cd ..

stage=0
input=ep
do_wer_tests=false

orig=$1
language=$2
pretrained=$3

export DATA_DIR=$orig/bert
tmp=$DATA_DIR/tmp
mkdir -p $tmp
d=$(date +'%Y%m%d')

error_calc_dir=utils
export SCRIPT_DIR=transformers/examples/token-classification #/home/staff/inga/transformers/examples/ner
export MAX_LENGTH=60
export MAX_SEQ_LENGTH=128
export BERT_MODEL=$pretrained
export OUTPUT_DIR=/work/inga/punctuation/NERtrans-out/$language-$pretrained-len$MAX_LENGTH-$d
export BATCH_SIZE=64
export NUM_EPOCHS=4
export SAVE_STEPS=1000
export SEED=42

if [ $stage -le -1 ]; then
    echo "Get the data on the right format"
    # 1. Put one word (+ a possible punct) per line,
    # 2. add a space between the punct token and the word,
    # 3-7. Change the punct symbols from the punctuator 2 look and map puncts to periods, commas and question marks
    # 8. Remove stuff stuck to the punct symbol, usually an apostrophe
    # 9. Keep only the first two columns (there can be more than one punctuation after a word)
    # 10. Remove lines containing only a label or empty
    # 11. Add O to empty slots in the 2nd column
    for i in train dev test; do
        (
            sed -re 's/[^A-ZÁÐÉÍÓÚÝÞÆÖa-záðéíóúýþæö0-9 .,:;\?\!$#@%&°\x27\/<>\-]/ /g' -e 's/ +/ /g' -e 's: ([\.,\?\!\:\;\-][A-Z]{4,}):\1:g' \
            < $orig/${input}.${i}.txt | tr ' ' '\n' \
            | sed -re 's:([\.,\?\!\:\;\-][A-Z]{4,}): \1:g' \
            -e 's:\;SEMICOLON|\-DASH|\:COLON:COMMA:g' \
            -e 's:\!EXCLAMATIONMARK:PERIOD:g' \
            -e 's:,COMMA:COMMA:g' \
            -e 's:\?QUESTIONMARK:QUESTIONMARK:g' \
            -e 's:\.PERIOD:PERIOD:g' \
            -e 's:(PERIOD|COMMA|QUESTIONMARK)[^ ]+:\1:g' \
            | cut -d' ' -f1,2 \
            | egrep -v '^ |^$' \
            | awk -F' ' '{if ($2 == "") print $1,"O"; else print $0}' \
            > $tmp/$i.txt.tmp
        ) &
    done
    wait;
    
    echo "Split the data into sequences of length $MAX_LENGTH."
    for i in train dev test; do
        awk -v m=$MAX_LENGTH ' {print;} NR % m == 0 { print ""; }' $tmp/$i.txt.tmp > ${DATA_DIR}/$i.txt &
    done
    wait;
    
    echo "Create a labels file"
    cat ${DATA_DIR}/{train.txt,dev.txt,test.txt} | cut -d " " -f 2 | egrep -v "^O?$"| sort | uniq > ${DATA_DIR}/labels.txt
fi

# echo "Fine-tune the model"
sbatch \
--job-name=${language}-bert --partition=longrunning \
--output=${DATA_DIR}/${language}_bert_$d.log \
--gres=gpu:3 --mem=32G --time=0-12:00 \
--wrap="srun \
python3 BERTbased/run_punctuation.py \
--data_dir ${DATA_DIR}/ \
--labels ${DATA_DIR}/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length $MAX_SEQ_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_device_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--evaluate_during_training \
--seed $SEED \
--fp16 \
--do_train \
--do_eval \
--do_predict"

# # Use the Punctuator2 F1-score calculator
# sed -re 's: O$::' -e 's:COMMA:,COMMA:' \
# -e 's:QUESTIONMARK:\?QUESTIONMARK:' \
# -e 's:PERIOD:.PERIOD:' \
# | tr '\n' ' ' | sed -r 's: +: :g' \
# > $OUTPUT_DIR/test_predictions_theano_style.txt

# echo 'Calculate F1-scores'
# python $error_calc_dir/error_calculator.py \
# <( sed -re 's/[^A-ZÁÐÉÍÓÚÝÞÆÖa-záðéíóúýþæö0-9 .,:;\?\!$#@%&°\x27\/<>\-]/ /g' -e 's/ +/ /g' $orig/${input}.test.txt) \
# $OUTPUT_DIR/test_predictions_theano_style.txt \
# > $OUTPUT_DIR/test_punct2error.txt

# NOTE! The following can't be run like this. Need to find another way
# if [ $do_wer_tests = "true" ]; then
#     echo 'Apply the model on text with different WER inserted'
#     bash wer-test.sh $input $DATA_DIR
# fi

exit 0;
