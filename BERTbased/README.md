The scripts in this directory fine tune a PyTorch transformer to a punctuation prediction task using the Hugging Face library
as explaned here: https://github.com/huggingface/transformers/tree/master/examples/token-classification
run_punctuation.py is barely changed from run_ner.py on that Github page.

The Icelandic Gigaword corpus data is obtained with ../process/rmh_subset_specific.ipynb and cleaned with rmh_data_cleaning.sh
The English Europarl data is obtained from https://www.statmt.org/europarl/ and cleaned with ../process/europarl_cleaning.sh
It is a bit confusing that the cleaning process is for punctuator 2. I should change that.

Install Hugging Face transformers:
cd ```
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
pip install -r ./examples/requirements.txt
```