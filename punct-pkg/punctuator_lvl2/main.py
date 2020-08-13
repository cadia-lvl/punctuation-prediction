# coding: utf-8

from punctuator_isl import api

import sys
import argparse
import urllib.request

ReadFile = argparse.FileType('r', encoding="utf-8")
WriteFile = argparse.FileType('w', encoding="utf-8")

parser = argparse.ArgumentParser(description="Punctuates Icelandic text")

parser.add_argument(
    'infile',
    nargs='?',
    type=ReadFile,
    default=sys.stdin,
    help="UTF-8 text file to punctuate"
)

parser.add_argument(
    'outfile',
    nargs='?',
    type=WriteFile,
    default=sys.stdout,
    help="UTF-8 output text file"
)

parser.add_argument(
    'modelfile',
    nargs='?',
    type=str,
    default="biRNN",
    help="Determines which models to use, the choices are between biRNN, BERT and seq2seq"

)

def main():

    args = parser.parse_args()
    input_path = args.infile
    output_path = args.outfile
    model_type = args.modelfile
    api.get_model(model_type)
    output_path.write(api.punctuate(input_path, args.modelfile, format='file'))

if __name__ == "__main__":
    main()
