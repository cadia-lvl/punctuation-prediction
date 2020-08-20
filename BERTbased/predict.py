import os
import sys
import logging
import unicodedata

import numpy as np
import torch
from transformers import AutoConfig, ElectraTokenizer, ElectraForTokenClassification

logging.basicConfig(level=logging.INFO)

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


def upcase_first_letter(s):
    return s[0].upper() + s[1:]


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


def punctuate(input_text, model_path):
    """Punctuate the input text with the ELECTRA model. Capitalize sentence beginnings."""

    config = AutoConfig.from_pretrained(model_path)
    tokenizer = ElectraTokenizer.from_pretrained(model_path)
    tokenizer.add_tokens(["<NUM>"])
    pytorch_model = ElectraForTokenClassification.from_pretrained(model_path)

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

    # split up long lines to not exceed the training sequence length
    n = 80
    text_to_punctuate = []
    if len(input_text) > n:
        line_part = [
            " ".join(input_text[x : x + n]) for x in range(0, len(input_text), n)
        ]
        text_to_punctuate.extend(line_part)
    elif len(input_text) == 0:
        pass
    else:
        text_to_punctuate.extend(input_text)

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


if __name__ == "__main__":

    if len(sys.argv) > 1:
        model_path = os.path.abspath(sys.argv[1])
    else:
        sys.exit(
            "'Model path' argument missing! Should be a directory containing pytorch_model.bin"
        )

    if len(sys.argv) > 2:
        input_file = os.path.abspath(sys.argv[4])
    else:
        sys.exit("An input textfile is required!")

    if len(sys.argv) > 3:
        output_file = os.path.abspath(sys.argv[5])
    else:
        sys.exit("An output textfile is required!")

    with open(input_file, "r") as f:
        # Clean the input file of non-printable characters and split on whitespace
        text = strip_string(f.read().replace("\n", " ")).split()

    with open(output_file, "x") as fout:
        fout.write(punctuate(text, model_path))
