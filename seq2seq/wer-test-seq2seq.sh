#!/bin/bash -e

# Make predictions with a fairseq seq2seq model on text sets with 5, 10, 15 of 20% WER inserted.
cd seq2seq
conda activate ptenv

orig=/work/helgasvala/tfpunctuationmars/tfpunctuation/Samanburður2604/bigdata/wer_testfiles
programdir=~/opt

wer_test=$1
wertestout=$2
input=$3
casing=${4:-}
if [ -n $casing ]; then
    ext=_$casing
fi

bindatadir=/work/inga/data/data-bin/${input}$ext.tokenized.nopuncts-${casing}puncts
modeldir=/work/inga/punctuation/fairseq-out/${input}$ext/checkpoints
tmp=$wer_test/tmp
prep=$wer_test/${input}${ext}.tokenized
toktmp=$prep/tmp
mkdir -p $tmp $prep $toktmp $wertestout

max_len=60
src=nopuncts
tgt=${casing}puncts

# Get all the test samples on the right format
for p in 5 10 15 20; do
    (
        tr '\n' ' ' < $orig/wer${p}perc.${input}.txt | awk -v m=$max_len '
    {
        n = split($0, a, " ")
        for (i=1; i<=n; i++)
            {
                printf "%s ",a[i]
                if (i % m == 0) {print ""}
            }
    }
        ' > $tmp/wer${p}perc.${input}.seq
        
        sed -re 's:[.,;:?\!-][A-Z]{4,}::g' \
        -e 's:.*:\L&:g' -e 's:<num>:NUM:g' \
        -e 's:^[^A-ZÁÐÉÍÓÚÝÞÆÖa-záðéíóúýþæö0-9]+::' \
        -e 's: $::' \
        -e 's:^(\bperiod\b|\bcomma\b|\bquestionmark\b|\bexclamationmark\b|\bdash\b|\bsemicolon\b|\bcolon\b) ::' \
        < $tmp/wer${p}perc.${input}.seq \
        > $wer_test/${input}.wer${p}perc.$src &
        
        sed -re 's:-DASH\b|\:COLON\b:,COMMA:g' \
        -e 's:;SEMICOLON\b|\!EXCLAMATIONMARK\b:.PERIOD:g' \
        -e 's: $::' -e 's:.*:\L&:g' -e 's:<num>:NUM:g' \
        -e 's:(\.period\b|,comma\b|\?questionmark\b):\U\1:g' \
        -e 's:^(\.PERIOD|,COMMA|\?QUESTIONMARK) ::' \
        -e 's:[.,?]([A-Z]{4,}):\1:g' \
        -e 's:^[^A-ZÁÐÉÍÓÚÝÞÆÖa-záðéíóúýþæö0-9]+::' \
        < $tmp/wer${p}perc.${input}.seq \
        > $wer_test/${input}.wer${p}perc.$tgt &
    ) &
done

SCRIPTS=$programdir/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl

l=en
for n in wer5perc wer10perc wer15perc wer20perc; do
    for p in $src $tgt; do
        cat $wer_test/$input.$n.$p | \
        perl $TOKENIZER -threads 8 -l $l > $toktmp/$n.tok.$p
        echo ""
    done
done

for n in wer5perc wer10perc wer15perc wer20perc; do
    perl $CLEAN -ratio 1.5 $toktmp/$n.tok $src $tgt $toktmp/$n.clean 1 175
done

for n in wer5perc wer10perc wer15perc wer20perc; do
    for p in $src $tgt; do
        srun python $BPEROOT/apply_bpe.py \
        -c $wer_test/../${input}${ext}.tokenized/code \
        --glossaries "PERIOD" "COMMA" "QUESTIONMARK" \
        < $toktmp/$n.clean.$p > $prep/$n.$p
    done
done

# Punctuate
mkdir -p $wertestout
for p in 5 10 15 20; do
    srun fairseq-interactive \
    $bindatadir \
    --input $prep/wer${p}perc.$src \
    --source-lang $src --target-lang $tgt \
    --path $modeldir/checkpoint_best.pt \
    --buffer-size 2000 --batch-size 128 \
    --beam 5 --remove-bpe \
    > $wertestout/${input}.wer${p}perc.out &
done


# Check if the model changes regular words and how many
for p in 5 10 15 20; do
    egrep "^D-" $wertestout/${input}.wer${p}perc.out | cut -f3- > $wertestout/${input}.wer${p}perc.punctuated
    
    wdiff -3 <(sed -r 's:PERIOD|COMMA|QUESTIONMARK::g' $toktmp/wer${p}perc.clean.$tgt) \
    <(sed -r 's:PERIOD|COMMA|QUESTIONMARK::g' $wertestout/${input}.wer${p}perc.punctuated) |egrep -v '=|^$' |wc -l
done

# Use the Punctuator2 F1-score calculator
for p in 5 10 15 20; do
    sed -e 's:COMMA:,COMMA:g' \
    -e 's:QUESTIONMARK:\?QUESTIONMARK:g' \
    -e 's:PERIOD:.PERIOD:g' \
    < $toktmp/wer${p}perc.clean.$tgt \
    | tr '\n' ' ' | sed -r 's: +: :g' \
    > $wertestout/${input}.wer${p}perc.target.punct2_style.txt &
    
    sed -e 's:COMMA:,COMMA:g' \
    -e 's:QUESTIONMARK:\?QUESTIONMARK:g' \
    -e 's:PERIOD:.PERIOD:g'\
    < $wertestout/${input}.wer${p}perc.punctuated \
    | tr '\n' ' ' | sed -r 's: +: :g' \
    > $wertestout/${input}.wer${p}perc.punctuated.punct2_style.txt
done

for p in 5 10 15 20; do
    python ../utils/error_calculator.py \
    $wertestout/${input}.wer${p}perc.target.punct2_style.txt $wertestout/${input}.wer${p}perc.punctuated.punct2_style.txt \
    > $wertestout/test_error_${p}wer.txt
done