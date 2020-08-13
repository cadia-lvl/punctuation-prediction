import numpy as np
import sys
import tensorflow as tf
from . import models
import urllib.request

def to_array(arr, dtype=np.int32):
    # minibatch of 1 sequence as column
    return np.array([arr], dtype=dtype).T

def convert_punctuation_to_readable(punct_token):
    if punct_token == models.SPACE:
        return " "
    else:
        return punct_token[0]
    
def capitalize_after_eos_token(token):
    if token[-2] in models.EOS_TOKENS:
        return token.title()
    else:
        return token

def get_model(model_type):
    if model_type == "biRNN":
        print("I'm here")
        model_url = "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/47/isl-model.pcl?sequence=3&isAllowed=y"
        vocab_url = "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/47/vocabulary?sequence=2&isAllowed=y"
        punct_url = "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/47/punctuations?sequence=1&isAllowed=y"
    elif model_type == "BERT":
        model_url = "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/47/isl-model.pcl?sequence=3&isAllowed=y"
        vocab_url = "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/47/vocabulary?sequence=2&isAllowed=y"
        punct_url = "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/47/punctuations?sequence=1&isAllowed=y"
    elif model_type == "seq2seq":
        model_url = "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/47/isl-model.pcl?sequence=3&isAllowed=y"
        vocab_url = "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/47/vocabulary?sequence=2&isAllowed=y"
        punct_url = "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/47/punctuations?sequence=1&isAllowed=y"
    else:
        sys.stderr.write("There is no matching model to the argument.")

    try:
        urllib.request.urlretrieve(model_url, 'isl-model_' + model_type + '.pcl')
        urllib.request.urlretrieve(vocab_url, 'vocabulary_' + model_type)
        urllib.request.urlretrieve(punct_url, 'punctuations_' + model_type)
    except:
        sys.stderr.write("The model could not be downloaded.")
        sys.exit(0)

def punctuate_text(word_vocabulary, punctuation_vocabulary, reverse_punctuation_vocabulary, input_text, model):

    i = 0
    output_list = []
    while True:

        subsequence = input_text[i:i+models.MAX_SEQUENCE_LEN]

        if len(subsequence) == 0:
            break

        converted_subsequence = [word_vocabulary.get(w, word_vocabulary[models.UNK]) for w in subsequence]

        y = tf.nn.softmax(model(to_array(converted_subsequence)))

        last_eos_idx = 0
        punctuations = []
        for y_t in y:

            p_i = np.argmax(tf.reshape(y_t, [-1]))
            punctuation = reverse_punctuation_vocabulary[p_i]

            punctuations.append(punctuation)

            if punctuation in models.EOS_TOKENS:
                last_eos_idx = len(punctuations) # we intentionally want the index of next element

        if subsequence[-1] == models.END:
            step = len(subsequence) - 1
        elif last_eos_idx != 0:
            step = last_eos_idx
        else:
            step = len(subsequence) - 1
    
        output_list.append(subsequence[0].title())
        for j in range(step):
            output_list.append(convert_punctuation_to_readable(punctuations[j]) + " " if punctuations[j] != (models.SPACE or models.END) else " ")
            output_list.append(subsequence[j+1].title() if punctuations[j] in models.EOS_TOKENS else subsequence[j+1])
            
        if subsequence[-1] == models.END:
            break

        i += step
        
    return output_list


def punctuate(input_text, model_type, format='inline'):
    model_file = "isl-model_" + model_type + ".pcl"
    vocab_file = "vocabulary_" + model_type
    punct_file = "punctuations_" + model_type
    vocab_len = len(models.read_vocabulary(models.WORD_VOCAB_FILE))
    x_len = vocab_len if vocab_len < models.MAX_WORD_VOCABULARY_SIZE else models.MAX_WORD_VOCABULARY_SIZE + models.MIN_WORD_COUNT_IN_VOCAB
    x = np.ones((x_len, models.MINIBATCH_SIZE)).astype(int)
    net, _ = models.load(model_file, x)
    word_vocabulary = net.x_vocabulary
    punctuation_vocabulary = net.y_vocabulary
    reverse_punctuation_vocabulary = {v:k for k,v in net.y_vocabulary.items()}

    if format=='file':
        lines = []
        for line in input_text:
            text = [w for w in line.split() if w not in punctuation_vocabulary] + [models.END]
            lines.append(text)    
        text_to_punctuate = [val for sublist in lines for val in sublist]
    else:
        text_to_punctuate = input_text.split() + [models.END]

    punctuated_list = punctuate_text(word_vocabulary, punctuation_vocabulary, reverse_punctuation_vocabulary, text_to_punctuate, net)

    for i in range(len(punctuated_list)-1,1,-1):
        if punctuated_list[i-1] == "</S>" and punctuated_list[i] in [". ", ", ", "? "]:
            punctuated_list.insert(i+1,"\n")
    
    punctuated_text = "".join([token for token in punctuated_list]).strip().replace(" </S>", "")

    return punctuated_text
