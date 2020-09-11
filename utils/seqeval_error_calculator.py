import logging
import sys
import os
import argparse
from seqeval.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""Computes and prints the overall classification error in terms of precision, recall and F-score.\n
        Used with input and output test files for the Hugging Face transformer.
        Usage: python perror_calculator.py <ground-truth-textfile> <predicted-text>\n
            E.g. python predict.py $DATA_DIR/test.txt $OUTPUT_DIR/test_predictions.txt
        """
    )
    parser.add_argument(
        "target_path", type=file_path, help="Ground truth text file",
    )
    parser.add_argument(
        "predicted_path", type=file_path, help="Model predictions",
    )
    parser.add_argument(
        "outfile", nargs="?", type=argparse.FileType("w"), default=sys.stdout
    )

    return parser.parse_args()


def file_path(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_file:{path} is not a valid file")


def compute_metrics(label_list, preds_list):
    return {
        "precision": precision_score(label_list, preds_list),
        "recall": recall_score(label_list, preds_list),
        "f1": f1_score(label_list, preds_list),
    }


def main():
    args = parse_arguments()

    label_list = [x.split(" ")[1] for x in open(args.target_path).readlines()]
    preds_list = [x.split(" ")[1] for x in open(args.predicted_path).readlines()]

    metrics = compute_metrics(label_list, preds_list)

    for key, value in metrics.items():
        logger.info("  %s = %s", key, value)
        args.outfile.write("%s = %s\n" % (key, value))


if __name__ == "__main__":

    main()
