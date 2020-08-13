#!/usr/bin/env bash
#
# Adapted from https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt14.sh
#
# Define the data directory
datadir=$1 # Created with rmh_data_cleaning.sh or europarl_cleaning.sh
input=$2 #rmh # name of data
casing=${3:-} # casing will stay empty if a 3rd argument is not passed #lc

src=nopuncts
tgt=${casing}puncts

prep=$datadir/$input.tokenized
tmp=$prep/tmp
programdir=~/opt

mkdir -p $tmp $prep $programdir

if [ ! -d "$programdir/mosesdecoder" ]; then
    echo 'Cloning Moses github repository (for tokenization scripts)...'
    git clone https://github.com/moses-smt/mosesdecoder.git $programdir
fi

if [ ! -d "$programdir/subword-nmt" ]; then
    echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
    git clone https://github.com/rsennrich/subword-nmt.git $programdir
fi

SCRIPTS=$programdir/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=$programdir/subword-nmt/subword_nmt
BPE_TOKENS=20000

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

echo "pre-processing train data..."
# NOTE! What language to use for the tokenizer
l=en # Use English for now and see if Haukur has an Icelandic tokenizer
for n in train dev test; do
    for p in $src $tgt; do
        cat $datadir/$input.$n.$p | \
        perl $TOKENIZER -threads 8 -l $l > $tmp/$n.tok.$p
        echo ""
    done
done

for n in train dev test; do
    perl $CLEAN -ratio 1.5 $tmp/$n.tok $src $tgt $tmp/$n.clean 1 175
done

TRAIN=$tmp/train.${tgt}-${src}
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.clean.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
# Tilgreina special tokens
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train dev test; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py \
        -c $BPE_CODE \
        --glossaries "PERIOD" "COMMA" "QUESTIONMARK" \
        < $tmp/$f.clean.$L > $prep/$f.$L &
    done
done
wait

exit 0