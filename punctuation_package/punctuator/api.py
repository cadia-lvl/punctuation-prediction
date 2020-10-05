import os
import sys
import logging
import unicodedata
from pathlib import Path
import json

import numpy as np
import tensorflow as tf
import requests
from transformers import AutoConfig, ElectraTokenizer, ElectraForTokenClassification

from .models import (
    SPACE,
    EOS_TOKENS,
    MAX_SEQUENCE_LEN,
    UNK,
    END,
    MAX_WORD_VOCABULARY_SIZE,
    MIN_WORD_COUNT_IN_VOCAB,
    MINIBATCH_SIZE,
    read_vocabulary,
    load,
)


logging.basicConfig(level=logging.WARNING)

# Electra had difficulties with non-printable symbols in text
# https://stackoverflow.com/questions/41757886/how-can-i-recognise-non-printable-unicode-characters-in-python
def strip_string(string):
    """Cleans a string based on a whitelist of printable unicode categories
  You can find a full list of categories here:
  http://www.fileformat.info/info/unicode/category/index.htm
  """
    letters = ("LC", "Ll", "Lm", "Lo", "Lt", "Lu")
    numbers = ("Nd", "Nl", "No")
    marks = ("Mc", "Me", "Mn")
    punctuation = ("Pc", "Pd", "Pe", "Pf", "Pi", "Po", "Ps")
    symbol = ("Sc", "Sk", "Sm", "So")
    space = ("Zs",)

    allowed_categories = letters + numbers + marks + punctuation + symbol + space

    return "".join([c for c in string if unicodedata.category(c) in allowed_categories])


def read_input(input_text, format):
    """Read the input file or string and clean of non-printable characters"""
    if format == "file":
        clean_text = strip_string(input_text.read().replace("\n", " "))
        input_text.close()
    else:
        clean_text = strip_string(input_text)
    return clean_text


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


def upcase_first_letter(s):
    return s[0].upper() + s[1:]


# Copy from nltk's downloader.py
def default_download_dir() -> str:
    # On Windows, use %APPDATA%
    if sys.platform == "win32" and "APPDATA" in os.environ:
        homedir = os.environ["APPDATA"]

    # Otherwise, install in the user's home directory.
    else:
        homedir = os.path.expanduser("~/")
        if homedir == "~/":
            raise ValueError("Could not find a default download directory")

    # append "punctuation_models" to the home directory
    return os.path.join(homedir, "punctuation_models")


def read_config_path():
    """Reads the path defined in path_config.json.
    Returns default path if null."""
    d = os.path.dirname(__file__)  # directory of script
    filename = f"{d}/path_config.json"

    with open(filename, "r") as config_file:
        config = json.load(config_file)

    if config["model_directory"] != "":
        return config["model_directory"]

    return default_download_dir()


def download_file(url, path_to_save):
    response = requests.get(url)
    if os.path.isfile(path_to_save):
        logging.info(f"Skip downloading {path_to_save} since already downloaded.")
    else:
        with open(path_to_save, "wb") as f:
            f.write(response.content)
        logging.info(f"Downloaded={path_to_save}")


def get_model(model_type, download_dir):
    """Get the model files from CLARIN"""

    model_dir = f"{download_dir}/{model_type}"
    try:
        if not os.path.exists(model_dir):
            Path(model_dir).mkdir(parents=True, exist_ok=True)
            print(f"Created the model directory: {model_dir}")
    except OSError:
        sys.exit(
            f"Fatal: The directory {model_dir} does not exist and cannot be created."
        )

    base_url = "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/52/"
    to_download = []
    if model_type == "biRNN":
        to_download.extend(
            [
                (
                    f"{base_url}Model_tf2_isl_big_1009_h256_lr0.02.pcl",
                    f"{model_dir}/model.pcl",
                ),
                (f"{base_url}vocabulary", f"{model_dir}/vocabulary"),
                (f"{base_url}punctuations", f"{model_dir}/punctuations",),
            ]
        )
    elif model_type == "ELECTRA":
        to_download.extend(
            [
                (f"{base_url}pytorch_model.bin", f"{model_dir}/pytorch_model.bin",),
                (f"{base_url}vocab.txt", f"{model_dir}/vocab.txt"),
                (f"{base_url}config.json", f"{model_dir}/config.json"),
                (
                    f"{base_url}tokenizer_config.json",
                    f"{model_dir}/tokenizer_config.json",
                ),
                (
                    f"{base_url}special_tokens_map.json",
                    f"{model_dir}/special_tokens_map.json",
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
    output_list = [input_text[0].title()]
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


def punctuate_biRNN(input_text, download_dir, model_type="biRNN", format="inline"):
    """Punctuate the input text with the Punctuator 2 model. Capitalize sentence beginnings."""
    get_model(model_type, download_dir)
    model_file = f"{download_dir}/{model_type}/model.pcl"
    vocab_file = f"{download_dir}/{model_type}/vocabulary"
    vocab_len = len(read_vocabulary(vocab_file))
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

    # Read the input
    text_to_punctuate = read_input(input_text, format).split() + [END]
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


def punctuate_electra(input_text, download_dir, model_type="ELECTRA", format="inline"):
    """Punctuate the input text with the ELECTRA model. Capitalize sentence beginnings."""
    get_model(model_type, download_dir)

    model_path = f"{download_dir}/{model_type}"
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = ElectraTokenizer.from_pretrained(model_path)
    tokenizer.add_tokens(["<NUM>"])
    pytorch_model = ElectraForTokenClassification.from_pretrained(model_path)
    pytorch_model.resize_token_embeddings(len(tokenizer))

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

    # Read the input and clean of non-printable characters
    input_list = read_input(input_text, format).split()

    # split up long lines to not exceed the training sequence length

    n = 60
    text_to_punctuate = []
    if len(input_list) > n:
        line_part = [
            " ".join(input_list[x : x + n]) for x in range(0, len(input_list), n)
        ]
        text_to_punctuate.extend(line_part)
    elif len(input_list) == 0:
        pass
    else:
        text_to_punctuate.append(" ".join(input_list))

    punctuated_text = []
    for t in text_to_punctuate:
        input_ids = tokenizer(t, return_tensors="pt")["input_ids"]
        tokens = tokenizer.tokenize(t)
        predictions = pytorch_model(input_ids)
        pred_ids = np.argmax(predictions[0].detach().numpy(), axis=2)[
            0
        ]  # Take the first matrix, since only have batch size 1
        predictions = [labels[pred_ids[i]] for i in range(1, len(pred_ids))]
        line_punctuated = iterate(tokens, predictions, eos_punct, punctuation_dict)
        punctuated_text.append(line_punctuated)

    return upcase_first_letter(" ".join(punctuated_text))


def iterate(tokens, predictions, eos_punct, punctuation_dict):
    """Iterate through a list of tokens and respective punctuation prediction per token
    and combine the two lists into one"""
    t2 = []
    p2 = []
    for t, p in zip(tokens, predictions):
        if t.startswith("##"):
            elem = t.split("##")[1]
            t2[-1] = t2[-1] + elem
        else:
            t2.append(t)
            p2.append(p)

    text_pred = []
    for t, p in zip(t2, p2):
        if text_pred != [] and text_pred[-1][-1] in eos_punct:
            text_pred.append(t.title())
        else:
            text_pred.append(t)
        if p == "O":
            pass
        else:
            text_pred[-1] = text_pred[-1] + punctuation_dict[p]

    return " ".join(text_pred)


def punctuate(input_path, model_type="biRNN", format="inline"):
    """Punctuate the input text"""

    download_dir = read_config_path()

    if model_type.lower() == "electra":
        punctuated_text = punctuate_electra(
            input_path, download_dir, model_type="ELECTRA", format=format
        )
    else:
        punctuated_text = punctuate_biRNN(
            input_path, download_dir, model_type="biRNN", format=format
        )

    return punctuated_text
