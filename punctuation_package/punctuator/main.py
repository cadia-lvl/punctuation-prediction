import os
import sys
import argparse
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


ReadFile = argparse.FileType("r", encoding="utf-8")
WriteFile = argparse.FileType("w", encoding="utf-8")

parser = argparse.ArgumentParser(description="Punctuates Icelandic text")

parser.add_argument(
    "infile",
    nargs="?",
    type=ReadFile,
    default=sys.stdin,
    help="UTF-8 text file to punctuate",
)

parser.add_argument(
    "outfile",
    nargs="?",
    type=WriteFile,
    default=sys.stdout,
    help="UTF-8 output text file",
)

group = parser.add_mutually_exclusive_group()
group.add_argument(
    "--birnn",
    help="Uses the bidirectional RNN model, Punctuator 2",
    action="store_true",
)
group.add_argument(
    "--electra", help="Uses an ELECTRA model, trained on Icelandic", action="store_true"
)


def main():

    args = parser.parse_args()
    input_path = args.infile
    output_path = args.outfile
    if args.electra:
        model_type = "ELECTRA"
    else:
        model_type = "biRNN"

    from .api import punctuate

    output_path.write(punctuate(input_path, model_type, format="file"))


if __name__ == "__main__":
    main()
