# coding: utf-8
from __future__ import division

import models, data, main

import sys
import codecs

import tensorflow as tf
import numpy as np

MAX_SUBSEQUENCE_LEN = 200

def to_array(arr, dtype=np.int32):
    # minibatch of 1 sequence as column
    return np.array([arr], dtype=dtype).T

def convert_punctuation_to_readable(punct_token):
    if punct_token == data.SPACE:
        return " "
    else:
        return punct_token[0]

def restore(output_file, text, word_vocabulary, reverse_punctuation_vocabulary, model):
    i = 0
    with codecs.open(output_file, 'w', 'utf-8') as f_out:
        while True:

            subsequence = text[i:i+MAX_SUBSEQUENCE_LEN]

            if len(subsequence) == 0:
                break

            converted_subsequence = [word_vocabulary.get(w, word_vocabulary[data.UNK]) for w in subsequence]

            y = predict(to_array(converted_subsequence), model)

            f_out.write(subsequence[0])

            last_eos_idx = 0
            punctuations = []
            for y_t in y:

                p_i = np.argmax(tf.reshape(y_t, [-1]))
                punctuation = reverse_punctuation_vocabulary[p_i]

                punctuations.append(punctuation)

                if punctuation in data.EOS_TOKENS:
                    last_eos_idx = len(punctuations) # we intentionally want the index of next element

            if subsequence[-1] == data.END:
                step = len(subsequence) - 1
            elif last_eos_idx != 0:
                step = last_eos_idx
            else:
                step = len(subsequence) - 1

            for j in range(step):
                f_out.write(" " + punctuations[j] + " " if punctuations[j] != data.SPACE else " ")
                if j < step - 1:
                    f_out.write(subsequence[1+j])

            if subsequence[-1] == data.END:
                break

            i += step

def predict(x, model):
    return tf.nn.softmax(net(x))

if __name__ == "__main__":

    if len(sys.argv) > 1:
        model_file = sys.argv[1]
    else:
        sys.exit("Model file path argument missing")

    if len(sys.argv) > 2:
        input_file = sys.argv[2]
    else:
        sys.exit("Input file path argument missing")

    if len(sys.argv) > 3:
        output_file = sys.argv[3]
    else:
        sys.exit("Output file path argument missing")

    vocab_len = len(data.read_vocabulary(data.WORD_VOCAB_FILE))
    x_len = vocab_len if vocab_len < data.MAX_WORD_VOCABULARY_SIZE else data.MAX_WORD_VOCABULARY_SIZE + data.MIN_WORD_COUNT_IN_VOCAB
    x = np.ones((x_len, main.MINIBATCH_SIZE)).astype(int)

    print("Loading model parameters...")
    net, _ = models.load(model_file, x)

    print("Building model...")

    word_vocabulary = net.x_vocabulary
    punctuation_vocabulary = net.y_vocabulary

    reverse_word_vocabulary = {v:k for k,v in word_vocabulary.items()}
    reverse_punctuation_vocabulary = {v:k for k,v in punctuation_vocabulary.items()}

    with codecs.open(input_file, 'r', 'utf-8') as f:
        input_text = f.read()

    if len(input_text) == 0:
        sys.exit("Input file empty.")

    text = [w for w in input_text.split() if w not in punctuation_vocabulary and w not in data.PUNCTUATION_MAPPING and not w.startswith(data.PAUSE_PREFIX)] + [data.END]
    pauses = [float(s.replace(data.PAUSE_PREFIX,"").replace(">","")) for s in input_text.split() if s.startswith(data.PAUSE_PREFIX)]

    restore(output_file, text, word_vocabulary, reverse_punctuation_vocabulary, net)
