#!/bin/bash -e

set -o pipefail

# Europarl data cleaning for punctuator2 training
# NOTE! Preprocessing for fairseq seq2seq transformer training added below

datadir=data/processed/ep
mkdir -p $datadir
if [ ! -d europarl ]; then
    wget -qO- http://hltshare.fbk.eu/IWSLT2012/training-monolingual-europarl.tgz data | tar xvz
    mv training-monolingual-europarl europarl
fi

conda activate tf21env

echo "Clean the data by removing and rewriting lines using sed regex"
# 1. All grep commands come from Ottokar. He ignores lines with these patterns
# 2. Sed: Remove content in parentheses and brackets
# 3. Remove remaining lines with (), [], {}
# 4. Change en dash to a hyphen
# 5. Remove hyphen after [;:,] and deal with multiple punctuation after words/numbers
# 6. Remove symbols other than letters or numbers at line beginnings and remove " and ' (latter around puncts)
# 7. Remove lines which don't contain letters, remove symbols listed in middle and change to one space between words.
grep -v " '[^ ]" data/europarl/europarl-v7.en | \
grep -v \'\ s\   | \
grep -v \'\ ll\  | \
grep -v \'\ ve\  | \
grep -v \'\ m\   | \
sed -re 's:\(+[^)]*?\)+: :g' -e 's:\[[^]]*?\]: :g' \
-e '/\[|\]|\{|\}|\(|\)/d' \
-e 's:–|--:-:g' \
-e 's/([;:,]) -/\1/g' -e 's:([^ .,:;?!-]+) ([.,:;?! -]+)([.,:;?!-]):\1 \3:g' \
-e 's:^[^A-Za-z0-9]+::' -e 's:"::g' \
-e '/^[^A-Za-z]*$/d' -e 's/[^A-Za-z0-9 .,:;\?\!$#@%&°\x27\/-]/ /g' -e 's/ +/ /g' \
> $datadir/ep_cleaned.txt

echo "Now the data can be formatted for training using preprocess.py"
srun python process/preprocess_en_lower.py \
$datadir/ep_cleaned.txt $datadir/ep_cleaned_formatted.txt \
&>$datadir/preprocess_data.log

echo "split it up into train, dev and test sets"
head -n -80000 $datadir/ep_cleaned_formatted.txt > $datadir/ep.train.txt
tail -n 80000 $datadir/ep_cleaned_formatted.txt > $datadir/devtest
head -n -40000 $datadir/devtest > $datadir/ep.dev.txt
tail -n 40000 $datadir/devtest > $datadir/ep.test.txt
echo "Cleaning up..."
rm -f $datadir/devtest
echo "Preprocessing done for punctuator 2 training. Now you can give the produced ./out dir as <data_dir> argument to data.py script for conversion and continue as described in the main README.md"

# Print info about the tokens in train, dev and test
for f in $datadir/ep.*.txt ; do grep -o .PERIOD $f | wc -l; done
for f in $datadir/ep.*.txt ; do grep -o ,COMMA $f | wc -l; done
for f in $datadir/ep.*.txt ; do grep -o ?QUESTIONMARK $f | wc -l; done
for f in $datadir/ep.*.txt ; do grep -o \!EXCLAMATIONMARK $f | wc -l; done
for f in $datadir/ep.*.txt ; do grep -o :COLON $f | wc -l; done
for f in $datadir/ep.*.txt ; do grep -o \;SEMICOLON $f | wc -l; done
for f in $datadir/ep.*.txt ; do grep -o "\-DASH" $f | wc -l; done

echo "Make a special dataset which fits for a seq2seq transformer training"
echo "and can be used to learn both punctuation and capitalization"
# For a transformer training I don't want to have the punctuation stuck against the punctuation token
# since that gets scrambled when tokenize. Remove the < and > symbols around number tokens
# I want to try a sequence to sequence training which also learns to capitalize the text so I use the
# original one which I have not changed the casing of
# Maybe I should have skipped running this through preprocess.py but since it is already done
# I will do the following:
echo "Remove line breaks and create segments of length $max_len"
tmp=$datadir/fairseq/tmp
mkdir -p $tmp
max_len=60 # for fairseq data
for n in train dev test; do
    tr '\n' ' ' < ep.$n.txt \
    | awk -v m=$max_len '
    {
        n = split($0, a, " ")
        for (i=1; i<=n; i++)
            {
                printf "%s ",a[i]
                if (i % m == 0) {print ""}
            }
    }
    ' > $tmp/ep.$n.puncts.seq &
done

for n in train dev test; do
    # Create an input text without punctuation tokens
    if [ ! -f "fairseq/ep.$n.nopuncts" ]; then
        sed -re 's:[.,;:?\!-][A-Z]{4,}::g' \
        -e 's:[<>]::g' \
        -e 's:^[^A-ZÁÐÉÍÓÚÝÞÆÖa-záðéíóúýþæö0-9]+::' -e 's: $::' \
        < $tmp/ep.$n.puncts.seq \
        > $datadir/fairseq/ep.$n.nopuncts &
    fi
    
    # Create an output text with 'no-special-symbol' punctuation tokens
    if [ ! -f "fairseq/ep.$n.puncts" ]; then
        sed -re 's:[.,;:?\!-]([A-Z]{4,}):\1:g' \
        -e 's:[<>]::g' \
        -e 's:^[^A-ZÁÐÉÍÓÚÝÞÆÖa-záðéíóúýþæö0-9]+::' \
        -e 's: $::' \
        -e 's:\bDASH\b|\bCOLON\b:COMMA:g' \
        -e 's:\bSEMICOLON\b|\bEXCLAMATIONMARK\b:PERIOD:g' \
        -e 's:^(PERIOD|COMMA|QUESTIONMARK) ::' \
        < $tmp/ep.$n.puncts.seq \
        > $datadir/fairseq/ep.$n.puncts &
    fi
done

echo "Preprocessing done for fairseq translational transformer training. Next step is running run-seq2seq-transformer.sh"

exit 0;
