#!/bin/bash -e

set -o pipefail

# RMH data cleaning for punctuation training
# NOTE! Created to fit with the punctuator 2 data processing
# but added afterwards cleaning for transformer training
# NOTE! A list of proper nouns needs to come from somewhere (words only used capitalized).
# Or a better true casing needs to be implemented
scriptdir=/home/staff/inga/h12/punctuation-detection
datadir=/work/inga/data/rmh_subset
tmp=$datadir/tmp
log=$datadir/log
punct2=$datadir/punctuator
fairseq=$datadir/fairseq
max_len=60 # for fairseq data

mkdir -p $tmp $log $punct2 $fairseq

conda activate tf21env

echo "Clean the data by removing and rewriting lines using sed regex"
# 1. Remove …„“”\"|«»‘*_<>●,, and trailing spaces
# 2. Remove lines which don't end with a EOS punctuation: [^\.\?\!]$
# 3. Remove lines containing ^, ¦, https (usually a long url follows)
#        or end with [ . or www ., and remove lines conaining three periods,
#        used to denote that the transcriber did not hear what was said, strange sentences often
# 4. Remove content in parentheses and brackets
# 5. Remove remaining parentheses, used in lists, e.g. a) bla, b) bla bla?
#         and remove remaining lines with (), [], {}
# 6. Rewrite simple urls, e.g. www.mbl.is and Vísir.is
# 7. Rewrite e-mail addresses, e.g. abc@abc.is -> abc @ abc punktur is
# 8. Rewrite dash and hyphens to "til" if between numbers or e.g. 2. apríl - 4. maí
# 9. Remove dash or hyphens if sandwitched between words e.g. Slysavarnarfélagið-Landsbjörg and before "og", "eða" and "né"
# 10. Change en dash to a hyphen
# 11. Remove hyphen after [;:,] and deal with multiple punctuation after words/numbers
# 12. Remove symbols other than letters or numbers at line beginnings
# 13. Remove lines which don't contain letters and change to one space between words.
for n in morgunbladid_subset ljosvakamidlar textasafn_arnastofnun; do
    sed -re 's:[…„“”\"\|«»‘*<>●]|::g' -e 's: ,, |_: :g' -e 's: +$::' \
    -e '/^.*?[^\.\?\!]$/d' \
    -e '/\^|¦|https|\[ \.$|www \.$|\.\.\./d' \
    -e 's:\(+[^)]*?\)+: :g' -e 's:\[[^]]*?\]: :g' \
    -e 's:(^| )(.{1,2}) \) :\1\2 :g' -e '/\[|\]|\{|\}|\(|\)/d' \
    -e 's:www\.([^\.]+).([^ ]+) :w w w \1 punktur \2 :g' -e 's:\.(is|com|net|org|edu|int|dk|co|no|se|fi|de)\b: punktur \1:g' \
    -e 's:\@([^\.]+).([^ ]+): @ \1 punktur \2:g' \
    -e 's:([0-9\.%]+)( ([a-záðéíóúýþæö]+ )?)[–-] ([0-9]):\1\2til \4:g' \
    -e 's:[–-]([^ ]): \1:g' -e 's: - (og|né|eða) : \1 :g' \
    -e 's:–:-:g' \
    -e 's/([;:,]) -/\1/g' -e 's:([^ .,:;?!-]+) ([.,:;?! -]+)([.,:;?!-]):\1 \3:g' \
    -e 's:^[^A-ZÁÐÉÍÓÚÝÞÆÖa-záðéíóúýþæö0-9]+::' \
    -e '/^[^A-ZÁÐÉÍÓÚÝÞÆÖa-záðéíóúýþæö]*$/d' -e 's/ +/ /g' \
    < $datadir/rmh_${n}.txt > $tmp/rmh_${n}_cleaned.txt
done

echo "Expand some abbreviations which don't have cases."
echo "Remove the periods from the rest"
# End with removing again lines not ending with an EOS punct.
for n in morgunbladid_subset ljosvakamidlar textasafn_arnastofnun; do
    # Start with expanding some abbreviations using regex
    sed -re 's:\ba\.m\.k ?\.:að minnsta kosti:g' \
    -e 's:\bág ?\.:ágúst:g' \
    -e 's:\bdes ?\.:desember:g' \
    -e 's:\bdr ?\.:doktor:g' \
    -e 's:\be\.t\.v ?\.:ef til vill:g' \
    -e 's:\bfeb ?\.:febrúar:g' \
    -e 's:\bfrh ?\.:framhald:g' \
    -e 's:\bfyrrv ?\.:fyrrverandi:g' \
    -e 's:\bheilbrrh ?\.:heilbrigðisráðherra:g' \
    -e 's:\biðnrh ?\.:iðnaðarráðherra:g' \
    -e 's:\binnanrrh ?\.:innanríkisráðherra:g' \
    -e 's:\bjan ?\.:janúar:g' \
    -e 's:\bkl ?\.:klukkan:g' \
    -e 's:\blandbrh ?\.:landbúnaðarráðherra:g' \
    -e 's:\bm\.a\.s ?\.:meira að segja:g' \
    -e 's:\bm\.a ?\.:meðal annars:g' \
    -e 's:\bmenntmrh ?\.:mennta og menningarmálaráðherra:g' \
    -e 's:\bm ?\.kr ?\.:millj kr:g' \
    -e 's:\bnk ?\.:næstkomandi:g' \
    -e 's:\bnóv ?\.:nóvember:g' \
    -e 's:\bnr ?\.:númer:g' \
    -e 's:\bnúv ?\.:núverandi:g' \
    -e 's:\bokt ?\.:október:g' \
    -e 's:\bo\.s\.frv ?\.:og svo framvegis:g' \
    -e 's:\bo\.þ\.h ?\.:og þess háttar:g' \
    -e 's:\bpr ?\.:per:g' \
    -e 's:\bsbr ?\.:samanber:g' \
    -e 's:\bsept ?\.:september:g' \
    -e 's:\bskv ?\.:samkvæmt:g' \
    -e 's:\bs\.s ?\.:svo sem:g' \
    -e 's:\bstk ?\.:stykki:g' \
    -e 's:\bt\.d ?\.:til dæmis:g' \
    -e 's:\bt\.a\.m ?\.:til að mynda:g' \
    -e 's:\bu\.þ\.b ?\.:um það bil:g' \
    -e 's:\butanrrh ?\.:utanríkisráðherra:g' \
    -e 's:\bviðskrh ?\.:viðskiptaráðherra:g' \
    -e 's:\bþáv ?\.:þáverandi:g' \
    -e 's:\b/þ\.e ?\.:það er:g' \
    -e 's:\bþús ?\.:þúsund:g' \
    -e 's:\bþ\.e\.a\.s ?\.:það er að segja:g' \
    -e 's:([A-ZÁÐÉÍÓÚÝÞÆÖa-záðéíóúýþæö])\.:\1:g' \
    -e '/^.*?[^\.\?\!]$/d' \
    < $tmp/rmh_${n}_cleaned.txt \
    > $tmp/rmh_${n}_cleaned_abbrexp.txt
done

echo "Create a single data set"
rm $tmp/ljosv_textas_morgunb.cleaned_subset.txt
for n in morgunbladid_subset ljosvakamidlar textasafn_arnastofnun; do
    cat $tmp/rmh_${n}_cleaned_abbrexp.txt >> $tmp/ljosv_textas_morgunb.cleaned_subset.txt
done

echo "Now the data can be formatted for training using preprocess_truecase.py"
srun python ${scriptdir}/process/preprocess_truecase.py \
$tmp/ljosv_textas_morgunb.cleaned_subset.txt $tmp/ljosv_textas_morgunb.cleaned_subset_formatted.txt \
&>$log/preprocess_data.log

echo "Remove duplicates"
awk '!x[$0]++' $tmp/ljosv_textas_morgunb.cleaned_subset_formatted.txt > $tmp/ljosv_textas_morgunb.cleaned_subset_formatted_uniq.txt

echo "Lowercase what comes after an EOS punctuation, unless it is an acronym"
file=ljosv_textas_morgunb.cleaned_subset_formatted_uniq
propernouns=/work/inga/data/rmh_subset/propernouns.tmp
# NOTE! I need to have a list of propernouns from somewhere. I extract it from a wordlist I have for Althingi
sed -r 's:^.*:\l&:' $propernouns |sort -u > $tmp/propernouns_lower.tmp
sed -re 's:^([A-ZÁÉÍÓÚÝÞÆÖ][a-záðéíóúýþæö ]):\l\1:' -e 's/:COLON ([A-ZÁÉÍÓÚÝÞÆÖ][a-záðéíóúýþæö ])/:COLON \l\1/g' \
< $tmp/$file.txt > $tmp/${file}_lower.tmp
egrep -o "^[a-záðéíóúýþæö]+|:COLON [a-záðéíóúýþæö]+" \
< $tmp/${file}_lower.tmp | sed -r 's/:COLON //' | sort -u \
> $tmp/lcwords.tmp
comm -12 <(sort -u lcwords.tmp) $tmp/propernouns_lower.tmp > $tmp/comm_proper.tmp
tr "\n" "|" < $tmp/comm_proper.tmp \
| sed '$s/|$//' | perl -pe "s:\|:\\\b\|\\\b:g" \
| sed 's:.*:\L&:' > $tmp/to_uppercase_pattern.tmp

# split creates files with names starting with an x
cd $tmp
split -n l/100 $tmp/${file}_lower.tmp
# Capitalize NOTE! I'm not sure what to do with the colons since the following word is not always capitalized
for f in x* ; do
    srun --mem 6G sed -re 's:^('$(cat $tmp/to_uppercase_pattern.tmp)'\b):\u\1:g' \
    -e 's:(\:COLON) ('$(cat $tmp/to_uppercase_pattern.tmp)'\b):\1 \u\2:g' \
    $tmp/$f > $tmp/${f}_truecase &
    cat x* >> $tmp/${file}_truecase.txt
done
rm x*

echo "split it up into train, dev and test sets"
# I want around 40k lines in each test set, without shuffling the lines.
# The data set contains around 8.3M lines so if I split the data set into 100 equal parts
# And select the top 800 lines from each, I should end up with dev and test files of correct size.
# Then I pipe the rest into a training file
split -n l/100 $tmp/${file}_truecase.txt

head -n400 x* | egrep -v "==>" | egrep -v "^ *$" > $punct2/rmh.dev.txt
for f in x* ; do
    head -n800 $f | tail -n +401 >> $punct2/rmh.test.txt
done
tail -n +801 x* | egrep -v "==>" | egrep -v "^ *$" > $punct2/rmh.train.txt
rm x*

# Print info about the tokens in train, dev and test
for f in $punct2/rmh.*; do grep -o .PERIOD $f | wc -l; done
for f in $punct2/rmh.*; do grep -o ,COMMA $f | wc -l; done
for f in $punct2/rmh.*; do grep -o ?QUESTIONMARK $f | wc -l; done
for f in $punct2/rmh.*; do grep -o \!EXCLAMATIONMARK $f | wc -l; done
for f in $punct2/rmh.*; do grep -o :COLON $f | wc -l; done
for f in $punct2/rmh.*; do grep -o \;SEMICOLON $f | wc -l; done
for f in $punct2/rmh.*; do grep -o "\-DASH" $f | wc -l; done

echo "Make a special dataset which fits for a seq2seq transformer training"
echo "and can be used to learn both punctuation and capitalization"
# For a transformer training I don't want to have the punctuation stuck against the punctuation token
# since that gets scrambled when tokenize. Remove the < and > symbols around number tokens
# I want to try a sequence to sequence training which also learns to capitalize the text so I use the
# original one which I have not changed the casing of
# Maybe I should have skipped running this through preprocess.py but since it is already done
# I will do the following:
split -n l/100 $tmp/${file}.txt

head -n400 x* | egrep -v "==>" | egrep -v "^ *$" > $tmp/rmh.dev.puncts.tmp
for f in x* ; do
    head -n800 $f | tail -n +401 >> $tmp/rmh.test.puncts.tmp
done
tail -n +801 x* | egrep -v "==>" | egrep -v "^ *$" > $tmp/rmh.train.puncts.tmp
rm x*

echo "Remove line breaks and create segments of length $max_len"
for n in train dev test; do
    tr '\n' ' ' < $tmp/rmh.$n.puncts.tmp \
    | awk -v m=$max_len '
    {
        n = split($0, a, " ")
        for (i=1; i<=n; i++)
            {
                printf "%s ",a[i]
                if (i % m == 0) {print ""}
            }
    }
    ' > $tmp/rmh.$n.puncts.seq &
done

for n in train dev test; do
    # Create a lowercased input text without punctuation tokens
    if [ ! -f "$fairseq/rmh.$n.nopuncts" ]; then
        sed -re 's:[.,;:?\!-][A-Z]{4,}::g' \
        -e 's:.*:\L&:g' -e 's:<num>:NUM:g' \
        -e 's:^[^A-ZÁÐÉÍÓÚÝÞÆÖa-záðéíóúýþæö0-9]+::' -e 's: $::' \
        < $tmp/rmh.$n.puncts.seq \
        > $fairseq/rmh.$n.nopuncts &
    fi
    # Create a truecased output text with 'no-special-symbol' punctuation tokens
    if [ ! -f "$fairseq/rmh.$n.puncts" ]; then
        sed -re 's:[.,;:?\!-]([A-Z]{4,}):\1:g' \
        -e 's:[<>]::g' \
        -e 's:^[^A-ZÁÐÉÍÓÚÝÞÆÖa-záðéíóúýþæö0-9]+::' \
        -e 's: $::' \
        -e 's:\bDASH\b|\bCOLON\b:COMMA:g' \
        -e 's:\bSEMICOLON\b|\bEXCLAMATIONMARK\b:PERIOD:g' \
        -e 's:^(PERIOD|COMMA|QUESTIONMARK) ::' \
        -e 's:(PERIOD|QUESTIONMARK) ([^ ]):\1 \u\2:g' \
        < $tmp/rmh.$n.puncts.seq \
        > $fairseq/rmh.n.puncts &
    fi
    
    if [ ! -f "$fairseq/rmh.$n.lcpuncts" ]; then
        sed -re 's:-DASH\b|\:COLON\b:,COMMA:g' \
        -e 's:;SEMICOLON\b|\!EXCLAMATIONMARK\b:.PERIOD:g' \
        -e 's: $::' -e 's:.*:\L&:g' -e 's:<num>:NUM:g' \
        -e 's:(\.period\b|,comma\b|\?questionmark\b):\U\1:g' \
        -e 's:^(\.PERIOD|,COMMA|\?QUESTIONMARK) ::' \
        -e 's:[.,?]([A-Z]{4,}):\1:g' \
        -e 's:^[^A-ZÁÐÉÍÓÚÝÞÆÖa-záðéíóúýþæö0-9]+::' \
        < $tmp/rmh.$n.puncts.seq \
        > $fairseq/rmh.$n.lcpuncts &
    fi
done

# Check the numbers for a seq2seq training sample
for f in $fairseq/rmh.train.puncts; do grep -o PERIOD $f | wc -l; done
for f in $fairseq/rmh.train.puncts; do grep -o COMMA $f | wc -l; done
for f in $fairseq/rmh.train.puncts; do grep -o QUESTIONMARK $f | wc -l; done

# ### For a 55M token subset I need to do the following:

# # 55M tokens means approx 2.7M lines. Hence
# # Mogginn: 0.27*2.7M = 730.000 lines
# # Ljosvakamidlar: 0.69*2.7M = 1.862.000 lines
# # Textasafn: 0.04*2.7M = 108000 lines
# # In the files textasafn comes first, then ljosvakamidlar and finally morgunbladid

# #Create a sample in these proportions:
# mkdir -p $punct2/sample55
# head -n 108000 $punct2/rmh.train.txt > $punct2/sample55/rmh.train.txt
# tail -n +350000 $punct2/rmh.train.txt | head -n 1862000 >> $punct2/sample55/rmh.train.txt
# tail -n 730000 $punct2/rmh.train.txt >> $punct2/sample55/rmh.train.txt

# # Had to redo from punctuator data in sample55:
# tmp=$fairseq/sample55/tmp
# mkdir -p $tmp
# for f in train dev test; do
#     tr '\n' ' ' < $punct2/sample55/rmh.${f}.txt | awk -v m=$max_len '
#     {
#         n = split($0, a, " ")
#         for (i=1; i<=n; i++)
#             {
#                 printf "%s ",a[i]
#                 if (i % m == 0) {print ""}
#             }
#     }
#     ' > $tmp/rmh.${f}.puncts.seq

#     sed -re 's:[.,;:?\!-][A-Z]{4,}::g' \
#     -e 's:.*:\L&:g' -e 's:<num>:NUM:g' \
#     -e 's:^[^A-ZÁÐÉÍÓÚÝÞÆÖa-záðéíóúýþæö0-9]+::' -e 's: $::' \
#     < $tmp/rmh.$f.puncts.seq \
#     > $fairseq/sample55/rmh.$f.nopuncts

#     sed -re 's:[.,;:?\!-]([A-Z]{4,}):\1:g' \
#     -e 's:[<>]::g' \
#     -e 's:^[^A-ZÁÐÉÍÓÚÝÞÆÖa-záðéíóúýþæö0-9]+::' \
#     -e 's: $::' \
#     -e 's:\bDASH\b|\bCOLON\b:COMMA:g' \
#     -e 's:\bSEMICOLON\b|\bEXCLAMATIONMARK\b:PERIOD:g' \
#     -e 's:^(PERIOD|COMMA|QUESTIONMARK) ::' \
#     -e 's:(PERIOD|QUESTIONMARK) ([^ ]):\1 \u\2:g' \
#     < $tmp/rmh.${f}.puncts.seq \
#     > $fairseq/sample55/rmh.${f}.puncts &

#     sed -re 's:-DASH\b|\:COLON\b:,COMMA:g' \
#     -e 's:;SEMICOLON\b|\!EXCLAMATIONMARK\b:.PERIOD:g' \
#     -e 's: $::' -e 's:.*:\L&:g' -e 's:<num>:NUM:g' \
#     -e 's:(\.period\b|,comma\b|\?questionmark\b):\U\1:g' \
#     -e 's:^(\.PERIOD|,COMMA|\?QUESTIONMARK) ::' \
#     -e 's:[.,?]([A-Z]{4,}):\1:g' \
#     -e 's:^[^A-ZÁÐÉÍÓÚÝÞÆÖa-záðéíóúýþæö0-9]+::' \
#     < $tmp/rmh.${f}.puncts.seq \
#     > $fairseq/sample55/rmh.${f}.lcpuncts &
# done

exit 0;
