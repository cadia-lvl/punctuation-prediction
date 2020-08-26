#!/bin/bash -e

set -o pipefail

# Run a hugging face ner transformer on a punctuation task as explained here:
# https://github.com/huggingface/transformers/tree/master/examples/token-classification
# The Icelandic Gigaword corpus data is obtained with ../process/rmh_subset_specific.ipynb
# and cleaned with rmh_data_cleaning.sh
# The English Europarl data is obtained from https://www.statmt.org/europarl/ and cleaned with ../process/europarl_cleaning.sh

stage=-1
do_wer_tests=false

if [ "$1" = "-h" ] || [ "$1" = "--help" ] || [ $# -lt 2 ]; then
    echo "This script fine tunes a PyTorch transformer to a punctuation prediction task using the Hugging Face library,"
    echo "as explaned here: https://github.com/huggingface/transformers/tree/master/examples/token-classification."
    echo "run_punctuation.py is barely changed from run_ner.py on that Github page."
    echo "It takes in a path to processed train, dev and test sets. These data sets have been processed using"
    echo "either rmh_data_cleaning.shor europarl_cleaning.sh in the process directory."
    echo "The script also takes in the language used, e.g. en for English and what pre-trained model to use."
    echo "It can either be a string signifying a model on Hugging-Face, e.g. bert-base-cased, or a path to a directory"
    echo "on your server containing a pretrained BERT like model, e.g. /work/inga/punctuation/electra."
    echo "The final input is an output directory, in which a subdirectory will be created withthe fine-tuned model."
    echo ""
    echo "Usage: $0 [options] <data-dir> <language string> <pretrained BERT model> <output-dir>"
    echo " e.g.: $0 --stage -1 /data/europarl/processed/en en out/punctuation/europarl"
    echo ""
    exit 1;
fi

orig=$1
language=$2 # in the case of Icelandic can be used for a dataset as well, e.g. rmh or althingi
pretrained=$3 # e.g. bert-base-multilingual-cased or path to a folder with the model
outdir=$4

export DATA_DIR=$orig/bert
tmp=$DATA_DIR/tmp
mkdir -p $tmp
d=$(date +'%Y%m%d')

export MAX_LENGTH=60
export MAX_SEQ_LENGTH=128
export BERT_MODEL=$pretrained #bert-base-multilingual-cased
export OUTPUT_DIR=$outdir/bert-out/$language-$(basename $pretrained)-len$MAX_LENGTH-$d
export BATCH_SIZE=16
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
            < $orig/${language}.${i}.txt | tr ' ' '\n' \
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

echo "Fine-tune the model"
sbatch \
--job-name=${language}-bert \
--output=${DATA_DIR}/${language}_bert_$d.log \
--gres=gpu:3 --mem=28G --time=0-12:00 \
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
--do_train \
--do_eval \
--do_predict"

# # Use the Punctuator2 F1-score calculator
# sed -re 's: O$::' -e 's:COMMA:,COMMA:' \
# -e 's:QUESTIONMARK:\?QUESTIONMARK:' \
# -e 's:PERIOD:.PERIOD:' \
# < $OUTPUT_DIR/test_predictions.txt \
# > $OUTPUT_DIR/test_predictions_theano_style.txt

# echo 'Calculate F1-scores'
# python utils/error_calculator.py \
# <( sed -re 's/[^A-ZÁÐÉÍÓÚÝÞÆÖa-záðéíóúýþæö0-9 .,:;\?\!$#@%&°\x27\/<>\-]/ /g' -e 's/ +/ /g' $orig/*.test.txt) \
# $OUTPUT_DIR/test_predictions_theano_style.txt \
# > $OUTPUT_DIR/test_punct2error.txt

# NOTE! The following can't be run like this. Need to find another way
# if [ $do_wer_tests = "true" ]; then
#     echo 'Apply the model on text with different WER inserted'
#     bash wer-test.sh $language $DATA_DIR $OUTPUT_DIR
# fi

exit 0;
