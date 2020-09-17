The scripts in this directory fine tune a PyTorch transformer to a punctuation prediction task using the Hugging Face library
as explaned here: https://github.com/huggingface/transformers/tree/master/examples/token-classification
run_punctuation.py is barely changed from run_ner.py on that Github page.

## DATA
The Icelandic Gigaword corpus data is obtained with `../process/rmh_subset_specific.ipynb` and cleaned with `../process/rmh_data_cleaning.sh`
The English Europarl data is obtained from https://www.statmt.org/europarl/ and cleaned with `../process/europarl_cleaning.sh`
It is a bit confusing that the cleaning process is for punctuator 2 and then adapted to the Hugging Face format. I should change that.

## INSTALLATION
Assume we are in a conda environment
`conda install pytorch`
Install Hugging Face transformers:
~~~~
pip install transformers
pip install seqeval
pip install git+https://github.com/fastai/fastprogress.git
~~~~

## RUN

Run with:
`run.sh [stage] <data-dir> <dataset-name> <outdir>`
E.g.:
`run.sh data/rmh_althingi rmh_althingi punctuation/bert-out/`

Where `<data-dir>` is the output from ```rmh_data_cleaning.sh``` or ```europarl_cleaning.sh```

## Predict
To insert punctuations into a text with no punctuations use `predict.py`. If you have a test set with punctuations in and want to score the prediction use `predict_for_scoring.py`. When training a model you will get a score when running `run.sh` but if you want to get a score on an existing model this is the easiest way.

Do:
`python predict_for_scoring.py <pytorch-model-dir> <input-file> <output-file>`
where `<input-file>` can contain punctuation tokens, as prepaired for a Hugging Face model training, i.e. as done in stage -1 in `run.sh`. 

## Scoring
If scoring the predictions of an existing model use either `utils/error_calculator.py --transformer <target-file> <predictions>` for a score like Punctuator 2 gives or `utils/seqeval_error_calculator.py <target-file> <predictions>`, for a score from seqeval, like what is given in the Hugging Face recipe.