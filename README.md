# punctuation-prediction
Support tools for punctuation prediction for ASR output. Three models are given or pointed to; a BERT-based Transformer, a seq2seq Transformer and a bidirectional RNN (Punctuator 2, www.github.com/ottokart/punctuator2)
in Tensorflow 2. 
Additionally, the code to preprocess text for the use of these models is given in the folder `process`.

The BERT based transformer is a token classifying transformer from https://github.com/huggingface/transformers, used here for punctuation prediction. 
The sequnece to sequence transformer comes from https://github.com/pytorch/fairseq and is based on the transformer described in the paper Attention is all you need. 
All we provide here for the transformers are 
1) data preprocessing scripts, to get the data on the right format for these models for the task of punctuation prediction, and
2) run files, where these models are trained for punctuation prediction.

## Requirements and Installation
- Python version >= 3.6
- An NVIDIA GPU and NCCL
- For the sequence to sequence model: PyTorch version >= 1.4.0
- For the BERT based token classifier and Punctuator 2: TensorFlow 2.0

Installation with the HuggingFace and Fairseq submodules:

git clone --recurse-submodules https://github.com/cadia-lvl/punctuation-prediction

## Licence
MIT License

Copyright (c) 2020 Language and Voice Lab
