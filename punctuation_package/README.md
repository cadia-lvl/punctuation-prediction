# README

# Punctuation Prediction 
A python package that punctuates Icelandic text. The input data is unpunctuated text and punctuated text is returned. The user can choose between two punctuation models, a bidirectional RNN ([Punctuator 2](www.github.com/ottokart/punctuator2)) in Tensorflow 2, and a pretrained ELECTRA Transformer, fine-tuned for punctuation prediction, [based on a Hugging Face NER recipe](https://github.com/huggingface/transformers/tree/master/examples/token-classification). The pretrained ELECTRA model was trained by Jón Friðrik Daðason on data from the Icelandic Gigaword corpus. Both punctuation models are trained/fine-tuned using Gigaword corpus data.

# Table of Contents
- [Installation](#installation)
- [Running](#running)
  * [Example](#example)
  * [Python module](#python-module)
    + [The punctuate function](#the-punctuate-function)
- [License](#license)
- [Authors/Credit](#authors-credit)
  * [Acknowledgements](#acknowledgements)

# Installation

To install, first create a conda environment:
```conda create --name {venv}```

Then activate it:
```conda activate {venv}```

Install the requirement(s):
```conda install tensorflow==2.1.0```
```pip install tensorflow==2.1.0```could also work, if you run into problems.

The transformer models are created with PyTorch, which is needed when reading the models:
```conda install pytorch```

To use the ELECTRA model one needs access to Hugging Face functions, installed with:

```pip install transformers```

Then finally, run:

```pip install punctuator-isl```

# Running

The program can be run either from a command line or from inside a python script. 

To run it on a command line:

```$ punctuate input.txt output.txt```

The default model is the *biRNN* model, you can also specify another model, e.g.:

```$ punctuate input.txt output.txt --electra```

The input uses `stdin` and the output `stdout`. Both files are encoded in UTF-8. 

Empty lines in the input are treated as sentence boundaries. 

Which of the two models to be used can be specified on the command line. The default is `biRNN`.

|Model|Description|
|---|---|
|biRNN|The Punctuator 2 model in Tensorflow.|
|ELECTRA|The ELECTRA Transformer (HuggingFace)|

For a short help message of how to use the package, type `punctuate -h` or `punctuate --help`.

The input text should be like directly from automatic speech recognition, without capitalizations or punctuations. 

The first time the program is run the punctuation models are downloaded into `punctuation_models`, in the user 
home directory for Linux and Mac users, and in APPDATA for Windows users. The user can put a new model path in 
`path_config.json` inside the punctuator directory is another location is desired.

## Example

In this case, the default model is used. An input string is specified and the punctuate function returns a punctuated string, words that appear after an end-of-sentence punctuation mark are capitalized.

```
$ echo "næsti fundur er fyrirhugaður í næstu viku að sögn kristínar jónsdóttur hópstjóra náttúruvárvöktunar hjá veðurstofu íslands verður áfram fylgst grannt með jarðhræringum á svæðinu" | punctuate
$ Næsti fundur er fyrirhugaður í næstu viku. Að sögn kristínar jónsdóttur, hópstjóra náttúruvöktunar hjá veðurstofu íslands, verður áfram fylgst grannt með jarðhræringum á svæðinu.
```


## Python module

### The punctuate function

```
from punctuator-is import punctuate

# A string to be punctuated
s = "næsti fundur er fyrirhugaður í næstu viku að sögn kristínar jónsdóttur hópstjóra náttúruvárvöktunar hjá veðurstofu íslands verður áfram fylgst grannt með jarðhræringum á svæðinu"

punctuated = punctuate(s, model='biRNN')

print(punctuated)
```

The program should output:
```
Næsti fundur er fyrirhugaður í næstu viku. Að sögn kristínar jónsdóttur, hópstjóra náttúruvöktunar hjá veðurstofu íslands, verður áfram fylgst grannt með jarðhræringum á svæðinu.
```

# License
This code is licensed under the MIT license.

# Authors/Credit
[Reykjavik University](www.ru.is)

Main authors: Helga Svala Sigurðardóttir - helgas@ru.is, Inga Rún Helgadóttir - ingarun@ru.is

## Acknowledgements

This project was funded by the Language Technology Programme for Icelandic 2019-2023. The programme, which is managed and coordinated by [Almannarómur](https://almannaromur.is/), is funded by the Icelandic Ministry of Education, Science and Culture.
