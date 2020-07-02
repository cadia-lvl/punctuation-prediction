The scripts in this directory train and test a fairseq sequence-to-sequence transformer that learns to insert punctuation.
We followed these instructions: https://github.com/pytorch/fairseq/blob/master/examples/translation/README.md#training-a-new-model

The Icelandic Gigaword corpus data is obtained with ../process/rmh_subset_specific.ipynb and cleaned with rmh_data_cleaning.sh

The English Europarl data is obtained and cleaned with ../process/europarl_cleaning.sh
