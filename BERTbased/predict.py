import os
import sys
import logging
import unicodedata
import argparse

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification

logging.basicConfig(level=logging.WARNING)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""Use a BERT like transformer, fine tuned for punctuation prediction, to insert punctuations into the input text.\n
        Usage: python predict.py <model-path> <input-file> <output-file>\n
            E.g. python predict.py out/punctuation/electra input.txt output.txt
        """
    )
    parser.add_argument(
        "model_path", type=dir_path, help="Path to punctuation model directory",
    )
    parser.add_argument(
        "infile", type=file_path, help="UTF-8 text file to punctuate",
    )
    parser.add_argument(
        "outfile", type=str, help="Punctuated output text file",
    )
    return parser.parse_args()


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


def file_path(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_file:{path} is not a valid file")


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
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.add_tokens(["<NUM>"])
    pytorch_model = AutoModelForTokenClassification.from_pretrained(model_path)

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


def main():
    args = parse_arguments()

    with open(args.infile, "r") as f:
        # Clean the input file of non-printable characters and split on whitespace
        text = strip_string(f.read().replace("\n", " ")).split()

    punctuated = punctuate(text, args.model_path)

    with open(args.outfile, "w") as fout:
        fout.write(punctuated)


if __name__ == "__main__":
    main()
