"""
Computes and prints the overall classification error and precision, recall, F-score over punctuations.
"""

__author__ = "Ottokar Tilk and Tanel Alumae. Adapted by Inga R. Helgadottir"

import os
import codecs
import argparse

from numpy import nan


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""Computes and prints the overall classification error and precision, recall, F-score over punctuations.\n
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
        "-t", "--transformer", action="store_true", help="The model is a transformer"
    )
    parser.add_argument(
        "-i", "--icelandic", action="store_true", help="The model is for Icelandic"
    )
    return parser.parse_args()


def file_path(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_file:{path} is not a valid file")


def punctuations(transformer, icelandic):
    # Can be used to estimate 2-class performance for example
    MAPPING = {}

    if transformer:
        PUNCTUATION_VOCABULARY = ["COMMA", "PERIOD", "QUESTIONMARK"]
        if icelandic:
            # Mapping to comma fits better for Icelandic, but to period for English
            PUNCTUATION_MAPPING = {
                "SEMICOLON": "COMMA",
                "COLON": "COMMA",
                "EXCLAMATIONMARK": "PERIOD",
                "DASH": "COMMA",
            }
        else:
            PUNCTUATION_MAPPING = {
                "SEMICOLON": "PERIOD",
                "COLON": "COMMA",
                "EXCLAMATIONMARK": "PERIOD",
                "DASH": "COMMA",
            }
    else:
        SPACE = "_SPACE"
        PUNCTUATION_VOCABULARY = [SPACE, ",COMMA", ".PERIOD", "?QUESTIONMARK"]
        if icelandic:
            PUNCTUATION_MAPPING = {
                ";SEMICOLON": ",COMMA",
                ":COLON": ",COMMA",
                "!EXCLAMATIONMARK": ".PERIOD",
                "-DASH": ",COMMA",
            }
        else:
            PUNCTUATION_MAPPING = {
                ";SEMICOLON": ".PERIOD",
                ":COLON": ",COMMA",
                "!EXCLAMATIONMARK": ".PERIOD",
                "-DASH": ",COMMA",
            }
    return MAPPING, PUNCTUATION_MAPPING, PUNCTUATION_VOCABULARY


def compute_error(target_paths, predicted_paths, transformer, icelandic):

    MAPPING, PUNCTUATION_MAPPING, PUNCTUATION_VOCABULARY = punctuations(
        transformer, icelandic
    )

    counter = 0
    total_correct = 0

    correct = 0.0
    substitutions = 0.0
    deletions = 0.0
    insertions = 0.0

    true_positives = {}
    false_positives = {}
    false_negatives = {}

    for target_path, predicted_path in zip(target_paths, predicted_paths):

        target_punctuation = " "
        predicted_punctuation = " "

        t_i = 0
        p_i = 0

        with codecs.open(target_path, "r", "utf-8") as target, codecs.open(
            predicted_path, "r", "utf-8"
        ) as predicted:

            target_stream = (
                target.read()
                .replace(" O\n", " ")
                .replace("\n", " ")
                .replace(" '", "'")
                .split()
            )
            predicted_stream = (
                predicted.read()
                .replace(" O\n", " ")
                .replace("\n", " ")
                .replace(" '", "'")
                .split()
            )

            while True:

                if (
                    PUNCTUATION_MAPPING.get(target_stream[t_i], target_stream[t_i])
                    in PUNCTUATION_VOCABULARY
                ):
                    # skip multiple consecutive punctuations
                    while (
                        PUNCTUATION_MAPPING.get(target_stream[t_i], target_stream[t_i])
                        in PUNCTUATION_VOCABULARY
                    ):
                        target_punctuation = PUNCTUATION_MAPPING.get(
                            target_stream[t_i], target_stream[t_i]
                        )
                        target_punctuation = MAPPING.get(
                            target_punctuation, target_punctuation
                        )
                        t_i += 1
                else:
                    target_punctuation = " "

                if predicted_stream[p_i] in PUNCTUATION_VOCABULARY:
                    predicted_punctuation = MAPPING.get(
                        predicted_stream[p_i], predicted_stream[p_i]
                    )
                    p_i += 1
                else:
                    predicted_punctuation = " "

                is_correct = target_punctuation == predicted_punctuation

                counter += 1
                total_correct += is_correct

                if predicted_punctuation == " " and target_punctuation != " ":
                    deletions += 1
                elif predicted_punctuation != " " and target_punctuation == " ":
                    insertions += 1
                elif (
                    predicted_punctuation != " "
                    and target_punctuation != " "
                    and predicted_punctuation == target_punctuation
                ):
                    correct += 1
                elif (
                    predicted_punctuation != " "
                    and target_punctuation != " "
                    and predicted_punctuation != target_punctuation
                ):
                    substitutions += 1

                true_positives[target_punctuation] = true_positives.get(
                    target_punctuation, 0.0
                ) + float(is_correct)
                false_positives[predicted_punctuation] = false_positives.get(
                    predicted_punctuation, 0.0
                ) + float(not is_correct)
                false_negatives[target_punctuation] = false_negatives.get(
                    target_punctuation, 0.0
                ) + float(not is_correct)

                assert (
                    target_stream[t_i] == predicted_stream[p_i]
                    or predicted_stream[p_i] == "<unk>"
                ), (
                    f"File: {target_path} \n"
                    + f"Error: {target_stream[t_i]} ({t_i}) != {predicted_stream[p_i]} ({p_i}) \n"
                    + f"Target context: {' '.join(target_stream[t_i - 2 : t_i + 2])} \n"
                    + f"Predicted context: {' '.join(predicted_stream[p_i - 2 : p_i + 2])}"
                )

                t_i += 1
                p_i += 1

                if t_i >= len(target_stream) - 1 and p_i >= len(predicted_stream) - 1:
                    break

    overall_tp = 0.0
    overall_fp = 0.0
    overall_fn = 0.0

    print("-" * 46)
    print(f"{'PUNCTUATION':<16} {'PRECISION':<9} {'RECALL':<9} {'F-SCORE':<9}")

    for p in PUNCTUATION_VOCABULARY:

        if p == "_SPACE":
            continue

        overall_tp += true_positives.get(p, 0.0)
        overall_fp += false_positives.get(p, 0.0)
        overall_fn += false_negatives.get(p, 0.0)

        punctuation = p
        precision = (
            (
                true_positives.get(p, 0.0)
                / (true_positives.get(p, 0.0) + false_positives[p])
            )
            if p in false_positives
            else nan
        )
        recall = (
            (
                true_positives.get(p, 0.0)
                / (true_positives.get(p, 0.0) + false_negatives[p])
            )
            if p in false_negatives
            else nan
        )
        f_score = (
            (2.0 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else nan
        )
        print(
            f"{punctuation:<16} {precision*100:<9.1f} {recall*100:<9.1f} {f_score*100:<9.1f}"
        )
    print("-" * 46)
    pre = overall_tp / (overall_tp + overall_fp) if overall_fp else nan
    rec = overall_tp / (overall_tp + overall_fn) if overall_fn else nan
    f1 = (2.0 * pre * rec) / (pre + rec) if (pre + rec) else nan
    print(f"{'Overall':<16} {pre*100:<9.1f} {rec*100:<9.1f} {f1*100:<9.1f}")
    print(
        "Err: {:.2f}".format(100.0 - float(total_correct) / float(counter - 1) * 100.0)
    )
    print(
        "SER: {:.1f}".format(
            (substitutions + deletions + insertions)
            / (correct + substitutions + deletions)
            * 100
        )
    )


def main():
    args = parse_arguments()

    transformer = False
    if args.transformer:
        transformer = True

    icelandic = False
    if args.icelandic:
        icelandic = True

    compute_error(
        [args.target_path], [args.predicted_path], transformer, icelandic,
    )


if __name__ == "__main__":
    main()
