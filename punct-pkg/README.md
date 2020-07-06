# README

# Punctuation Prediction 
A python package that punctuates Icelandic text. The input data is unpunctuated text and punctuated text is returned. The user can choose between three different punctuation models, a BERT-based Transformer, a seq2seq Transformer and a bidirectional RNN ([Punctuator 2](www.github.com/ottokart/punctuator2)) in Tensorflow 2. We've written a paper on this topic and sent it to Interspeech. TODO: Add link to paper if it gets published.

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

Pip install the requirement(s):
```pip install tensorflow==2.1.0```
If pip, which the Tensorflow site recommends, for some reason doesn't work, you can try conda:
```conda install tensorflow==2.1.0```

Then finally, run:

```pip install lvl-punctuator```

# Running

The program can be run either from a command line or from inside a python script. 

To run it on a command line:

```$ punctuate input.txt output.txt```

The default model is the *biRNN* model, you can also specify another model, e.g.:

```$ punctuate input.txt output.txt "BERT"```

The input uses `stdin` and the output `stdout`. Both files are encoded in UTF-8. 

Empty lines in the input are treated as sentence boundaries. 

Which of the three models to be used can be specified on the command line. The default is `biRNN`.

|Model|Description|
|---|---|
|biRNN|The Punctuator 2 model in Tensorflow.|
|BERT|The BERT-based Transformer (HuggingFace)|
|seq2seq|The seq2seq Transformer (Fairseq)|

For a short help message of how to use the package, type `punctuate -h` or `punctuate --help`.

The input text should be like directly from automatic speech recognition, without capitalizations or punctuations. 

## Example

In this case, the default model is used. An input string is specified and the punctuate function returns a punctuated string, words that appear after an end-of-sentence punctuation mark are capitalized.

```
$ echo "næsti fundur er fyrirhugaður í næstu viku að sögn kristínar jónsdóttur hópstjóra náttúruvárvöktunar hjá veðurstofu íslands verður áfram fylgst grannt með jarðhræringum á svæðinu" | punctuate
$ Næsti fundur er fyrirhugaður í næstu viku. Að sögn kristínar jónsdóttur, hópstjóra náttúruvöktunar hjá veðurstofu íslands, verður áfram fylgst grannt með jarðhræringum á svæðinu.
```
The --seq2seq model also capitalizes proper nouns. This is an example output of that:
```
$ Næsti fundur er fyrirhugaður í næstu viku. Að sögn Kristínar Jónsdóttur, hópstjóra náttúruvöktunar hjá Veðurstofu Íslands, verður áfram fylgst grannt með jarðhræringum á svæðinu.
```

## Python module

### The punctuate function

```
from punctuator-is import punctuate, get_model

#This downloads the model that the user wants to use:
get_model('biRNN')


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
