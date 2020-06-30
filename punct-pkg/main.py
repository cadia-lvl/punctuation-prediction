# coding: utf-8

from __future__ import division

import punctuator_isl
from punctuator_isl import tools
import clarinmappa

import sys
from io import open
import argparse

import tensorflow as tf
import numpy as np
import argparse

ReadFile = argparse.FileType('r', encoding="utf-8")
WriteFile = argparse.FileType('w', encoding="utf-8")

parser = argparse.ArgumentParser(description="Punctuates Icelandic text")

parser.add_argument(
    'infile',
    nargs='?',
    type=ReadFile,
    default=sys.stdin,
    help="UTF-8 text file to punctuate",
)
parser.add_argument(
    'outfile',
    nargs='?',
    type=WriteFile,
    default=sys.stdout,
    help="UTF-8 output text file"
)

def to_array(arr, dtype=np.int32):
    # minibatch of 1 sequence as column
    return np.array([arr], dtype=dtype).T

def convert_punctuation_to_readable(punct_token):
    if punct_token == tools.SPACE:
        return " "
    else:
        return punct_token[0]
    
def capitalize_after_eos_token(token):
    if token[-2] in tools.EOS_TOKENS:
        return token.title()
    else:
        return token

def punctuate_text(word_vocabulary, punctuation_vocabulary, reverse_punctuation_vocabulary, input_text, model):

    i = 0
    output_list = []
    while True:

        subsequence = input_text[i:i+tools.MAX_SEQUENCE_LEN]

        if len(subsequence) == 0:
            break

        converted_subsequence = [word_vocabulary.get(w, word_vocabulary[tools.UNK]) for w in subsequence]

        y = tf.nn.softmax(model(to_array(converted_subsequence)))

        last_eos_idx = 0
        punctuations = []
        for y_t in y:

            p_i = np.argmax(tf.reshape(y_t, [-1]))
            punctuation = reverse_punctuation_vocabulary[p_i]

            punctuations.append(punctuation)

            if punctuation in tools.EOS_TOKENS:
                last_eos_idx = len(punctuations) # we intentionally want the index of next element

        if subsequence[-1] == tools.END:
            step = len(subsequence) - 1
        elif last_eos_idx != 0:
            step = last_eos_idx
        else:
            step = len(subsequence) - 1
    
        output_list.append(subsequence[0].title())
        for j in range(step):
            output_list.append(convert_punctuation_to_readable(punctuations[j]) + " " if punctuations[j] != (tools.SPACE or tools.END) else " ")
            output_list.append(subsequence[j+1].title() if punctuations[j] in tools.EOS_TOKENS else subsequence[j+1])
            
        if subsequence[-1] == tools.END:
            break

        i += step
        
    return output_list

def punctuate(input_text, format='inline'):
    model_file = "isl-model.pcl"
    vocab_len = vocab_len = len(tools.read_vocabulary(tools.WORD_VOCAB_FILE))
    x_len = vocab_len if vocab_len < tools.MAX_WORD_VOCABULARY_SIZE else tools.MAX_WORD_VOCABULARY_SIZE + tools.MIN_WORD_COUNT_IN_VOCAB
    x = np.ones((x_len, tools.MINIBATCH_SIZE)).astype(int)
    net, _ = tools.load(model_file, x)
    word_vocabulary = net.x_vocabulary
    punctuation_vocabulary = net.y_vocabulary
    reverse_punctuation_vocabulary = {v:k for k,v in net.y_vocabulary.items()}

    if format=='file':
        lines = []
        for line in input_text:
            text = [w for w in line.split() if w not in punctuation_vocabulary] + [tools.END]
            lines.append(text)    
        text_to_punctuate = [val for sublist in lines for val in sublist]
    else:
        text_to_punctuate = input_text.split() + [tools.END]

    punctuated_list = punctuate_text(word_vocabulary, punctuation_vocabulary, reverse_punctuation_vocabulary, text_to_punctuate, net)

    for i in range(len(punctuated_list)-1,1,-1):
        if punctuated_list[i-1] == "</S>" and punctuated_list[i] in [". ", ", ", "? "]:
            punctuated_list.insert(i+1,"\n")
    
    punctuated_text = "".join([token for token in punctuated_list]).strip().replace(" </S>", "")

    return punctuated_text

def main():
    args = parser.parse_args()
    input_path = args.infile
    print(input_path)
    output_path = args.outfile
    output_path.write(punctuate(input_path, format='file'))

if __name__ == "__main__":
    main()
