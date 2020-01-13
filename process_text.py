#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
# The following code is from Otto Kartoman, to mark punctuations in text
# It's available on: https://github.com/ottokart/punctuator2/blob/master/example/dont_run_me_run_the_other_script_instead.py
# coding: utf-8

from __future__ import division, print_function
from nltk.tokenize import word_tokenize

import nltk
import os
from io import open
import re
import sys

nltk.download('punkt')

NUM = '<NUM>'

EOS_PUNCTS = {".": ".PERIOD", "?": "?QUESTIONMARK", "!": "!EXCLAMATIONMARK"}
INS_PUNCTS = {",": ",COMMA", ";": ";SEMICOLON", ":": ":COLON", "-": "-DASH"}

forbidden_symbols = re.compile(r"[\[\]\(\)\/\\\>\<\=\+\_\*]")
numbers = re.compile(r"\d")
multiple_punct = re.compile(r'([\.\?\!\,\:\;\-])(?:[\.\?\!\,\:\;\-]){1,}')

is_number = lambda x: len(numbers.sub("", x)) / len(x) < 0.6

def untokenize(line):
    return line.replace(" ruv ", "")

def process_line(line):

    tokens = word_tokenize(line)
    output_tokens = []

    for token in tokens:

        if token in INS_PUNCTS:
            output_tokens.append(INS_PUNCTS[token])
        elif token in EOS_PUNCTS:
            output_tokens.append(EOS_PUNCTS[token])
        elif is_number(token):
            output_tokens.append(NUM)
        else:
            output_tokens.append(token.lower())

    return " ".join(output_tokens) + " "

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

# Get the contents of a file into the correct format for processing
def read_file_string(src):
    with open(src, 'r') as file:
        lines = file.read()
    splitLines = lines.split('\n')
    return [x.split('\t') for x in splitLines]

# Now we process the input text according to the previous script
if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_string = read_file_string(sys.argv[1])
    else:
        print("There is no specified text to process")

    processed_text = [process_line(elem) for elem in file_string]

    # Write it to train, dev, and test files.
    if len(sys.argv)  == 4:
        train_text, tmp_text = train_test_split(processed_text, test_size=sys.argv[2], random_state=42)
        dev_text, test_text = train_test_split(tmp_text, test_size=sys.argv[3] random_state=42)
    else:
        train_text, tmp_text = train_test_split(processed_text, test_size=0.2, random_state=42)
        dev_text, test_text = train_test_split(tmp_text, test_size=0.5 random_state=42)

    with open("processed_text.train.txt", 'w', encoding='utf-8') as train_file:
        for item in train_text:
            train_file.write("%s\n" % item)

    with open("processed_text.dev.txt", 'w', encoding='utf-8') as dev_file:
        for item in dev_text:
            dev_file.write("%s\n" % item)

    with open("processed_text.test.txt", 'w', encoding='utf-8') as test_file:
        for item in test_text:
            test_file.write("%s\n" % item)
