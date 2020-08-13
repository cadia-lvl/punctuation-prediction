import os
import sys
import logging

import numpy as np
import tensorflow as tf
import requests
from transformers import AutoConfig, ElectraTokenizer, ElectraForTokenClassification

from models import (
    SPACE,
    EOS_TOKENS,
    MAX_SEQUENCE_LEN,
    UNK,
    END,
    WORD_VOCAB_FILE,
    PUNCT_VOCAB_FILE,
    MAX_WORD_VOCABULARY_SIZE,
    MIN_WORD_COUNT_IN_VOCAB,
    MINIBATCH_SIZE,
    read_vocabulary,
    load,
)


logging.basicConfig(level=logging.INFO)


def to_array(arr, dtype=np.int32):
    # minibatch of 1 sequence as column
    return np.array([arr], dtype=dtype).T


def convert_punctuation_to_readable(punct_token):
    if punct_token == SPACE:
        return " "
    else:
        return punct_token[0]


def capitalize_after_eos_token(token):
    if token[-2] in EOS_TOKENS:
        return token.title()
    else:
        return token


def download_file(url, path_to_save):
    response = requests.get(url)
    if os.path.isfile(path_to_save):
        logging.info(f"Skip downloading {path_to_save} since already downloaded.")
    else:
        with open(path_to_save, "wb") as f:
            f.write(response.content)
        logging.info(f"Downloaded={path_to_save}")


def get_model(model_type):
    try:
        os.mkdir(model_type)
        print(f"Created {model_type}-folder")
    except:
        if os.path.exists(model_type):
            pass
    base_url = "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/48/"
    to_download = []
    if model_type == "biRNN":
        to_download.extend(
            [
                (f"{base_url}isl-model.pcl", model_type + "/model.pcl"),
                (f"{base_url}vocabulary", model_type + "/vocabulary"),
                (f"{base_url}punctuations", model_type + "/punctuations"),
            ]
        )
    elif model_type == "electra":
        to_download.extend(
            [
                (f"{base_url}pytorch_model.bin", model_type + "/pytorch_model.bin"),
                (f"{base_url}vocab.txt", model_type + "/vocab.txt"),
                (f"{base_url}config.json", model_type + "/config.json"),
                (
                    f"{base_url}tokenizer_config.json",
                    model_type + "/tokenizer_config.json",
                ),
                (
                    f"{base_url}special_tokens_map.json",
                    model_type + "/special_tokens_map.json",
                ),
            ]
        )
    else:
        sys.stderr.write("There is no matching model to the argument.")

    for download_args in to_download:
        try:
            download_file(*download_args)
        except:
            logging.error("Could not download file.")
            sys.exit(0)

def punctuate_text(
    word_vocabulary,
    punctuation_vocabulary,
    reverse_punctuation_vocabulary,
    input_text,
    model,
):

    i = 0
    output_list = []
    while True:

        subsequence = input_text[i : i + MAX_SEQUENCE_LEN]

        if len(subsequence) == 0:
            break

        converted_subsequence = [
            word_vocabulary.get(w, word_vocabulary[UNK]) for w in subsequence
        ]

        y = tf.nn.softmax(model(to_array(converted_subsequence)))

        last_eos_idx = 0
        punctuations = []
        for y_t in y:

            p_i = np.argmax(tf.reshape(y_t, [-1]))
            punctuation = reverse_punctuation_vocabulary[p_i]

            punctuations.append(punctuation)

            if punctuation in EOS_TOKENS:
                # we intentionally want the index of next element
                last_eos_idx = len(punctuations)

        if subsequence[-1] == END:
            step = len(subsequence) - 1
        elif last_eos_idx != 0:
            step = last_eos_idx
        else:
            step = len(subsequence) - 1

        output_list.append(subsequence[0].title())
        for j in range(step):
            output_list.append(
                convert_punctuation_to_readable(punctuations[j]) + " "
                if punctuations[j] != (SPACE or END)
                else " "
            )
            output_list.append(
                subsequence[j + 1].title()
                if punctuations[j] in EOS_TOKENS
                else subsequence[j + 1]
            )

        if subsequence[-1] == END:
            break

        i += step

    return output_list


def punctuate_biRNN(input_text, format="inline"):
    get_model("biRNN")
    model_file = "biRNN" + "/model.pcl"
    vocab_len = len(read_vocabulary(WORD_VOCAB_FILE))
    x_len = (
        vocab_len
        if vocab_len < MAX_WORD_VOCABULARY_SIZE
        else MAX_WORD_VOCABULARY_SIZE + MIN_WORD_COUNT_IN_VOCAB
    )
    x = np.ones((x_len, MINIBATCH_SIZE)).astype(int)
    net, _ = load(model_file, x)
    word_vocabulary = net.x_vocabulary
    punctuation_vocabulary = net.y_vocabulary
    reverse_punctuation_vocabulary = {v: k for k, v in net.y_vocabulary.items()}

    if format == "file":
        lines = []
        for line in input_text:
            text = [w for w in line.split() if w not in punctuation_vocabulary] + [END]
            lines.append(text)
        text_to_punctuate = [val for sublist in lines for val in sublist]
    else:
        text_to_punctuate = input_text.split() + [END]

    punctuated_list = punctuate_text(
        word_vocabulary,
        punctuation_vocabulary,
        reverse_punctuation_vocabulary,
        text_to_punctuate,
        net,
    )

    for i in range(len(punctuated_list) - 1, 1, -1):
        if punctuated_list[i - 1] == "</S>" and punctuated_list[i] in [
            ". ",
            ", ",
            "? ",
        ]:
            punctuated_list.insert(i + 1, "\n")

    punctuated_text = (
        "".join([token for token in punctuated_list]).strip().replace(" </S>", "")
    )

    return punctuated_text


def punctuate_electra(input_text, format="inline"):
    get_model("electra")

    config = AutoConfig.from_pretrained("electra")
    tokenizer = ElectraTokenizer.from_pretrained("electra")
    tokenizer.add_tokens(["<NUM>"])
    pytorch_model = ElectraForTokenClassification.from_pretrained("electra")

    punctuation_dict = {
        "COMMA": ",",
        "PERIOD": ".",
        "QUESTIONMARK": "?",
        "EXCLAMATIONMARK": "!",
        "COLON": ":",
        "SEMICOLON": ";",
        "DASH": "-",
    }
    eos_punct = [".", "?", "!"]

    labels = config.id2label
    # logging.info(labels)

    text_to_punctuate = []
    for line in input_text:
        l = line.split()
        n = 80
        if len(l) > n:
            line_part = [" ".join(l[x : x + n]) for x in range(0, len(l), n)]
            text_to_punctuate.extend(line_part)
        else:
            text_to_punctuate.append(line)

    punctuated_text = []
    for line in text_to_punctuate:
        input_ids = tokenizer(line, return_tensors="pt")["input_ids"]
        tokens = tokenizer.tokenize(line)
        predictions = pytorch_model(input_ids)
        pred_ids = np.argmax(predictions[0].detach().numpy(), axis=2)[0]
        predictions = [labels[pred_ids[i]] for i in range(1, len(pred_ids))]
        # logging.info(input_ids)
        logging.info(tokens)
        logging.info(predictions)
        line_punctuated = iterate(tokens, predictions, eos_punct, punctuation_dict)
        punctuated_text.append(line_punctuated)

    # input_ids = tokenizer.encode(
    #   text_to_punctuate, add_special_tokens=True, return_tensors="pt"
    # )
    # input_ids = tokenizer(text_to_punctuate, return_tensors="pt")["input_ids"]
    # logging.info(predictions[0].detach().numpy())
    # logging.info(preds_list)

    return " ".join(punctuated_text)


def iterate(tokens, predictions, eos_punct, punctuation_dict):
    t2 = []
    p2 = []
    for t, p in zip(tokens, predictions):
        if t.startswith("##"):
            elem = t.split("##")[1]
            t2[-1] = t2[-1] + elem
        else:
            t2.append(t)
            p2.append(p)

    logging.info(t2)
    logging.info(p2)

    text_pred = []
    for t, p in zip(t2, p2):
        if text_pred == [] or text_pred[-1][-1] in eos_punct:
            text_pred.append(t.title())
        else:
            text_pred.append(t)
        if p == "O":
            pass
        else:
            text_pred[-1] = text_pred[-1] + punctuation_dict[p]

    logging.info(" ".join(text_pred))
    return " ".join(text_pred)


def punctuate(input_text, model_type="biRNN", format="inline"):
    if model_type == "biRNN":
        punctuated_text = punctuate_biRNN(input_text, format="inline")
        logging.info(punctuated_text)
    elif model_type == "electra":
        punctuated_text = punctuate_electra(input_text, format="inline")
        logging.info(punctuated_text)

    return punctuated_text
