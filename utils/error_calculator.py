"""
Computes and prints the overall classification error and precision, recall, F-score over punctuations.
"""

__author__ = 'Ottokar Tilk and Tanel Alumae. Adapted by Inga R. Helgadottir'

import sys
import codecs

from numpy import nan

# Can be used to estimate 2-class performance for example
MAPPING = {}
# MAPPING = {"!EXCLAMATIONMARK": ".PERIOD", "?QUESTIONMARK": ".PERIOD", ":COLON": ".PERIOD", ";SEMICOLON": ".PERIOD"}

# From data.py in Punctuator 2. Move it here since using this file with transformer results
SPACE = "_SPACE"
PUNCTUATION_VOCABULARY = [SPACE, ",COMMA",
                          ".PERIOD", "?QUESTIONMARK"]
PUNCTUATION_MAPPING = {";SEMICOLON": ".PERIOD", ":COLON": ",COMMA",
                       "!EXCLAMATIONMARK": ".PERIOD", "-DASH": ",COMMA"}


def compute_error(target_paths, predicted_paths):
    counter = 0
    total_correct = 0

    correct = 0.
    substitutions = 0.
    deletions = 0.
    insertions = 0.

    true_positives = {}
    false_positives = {}
    false_negatives = {}

    for target_path, predicted_path in zip(target_paths, predicted_paths):

        target_punctuation = " "
        predicted_punctuation = " "

        t_i = 0
        p_i = 0

        with codecs.open(target_path, 'r', 'utf-8') as target, codecs.open(predicted_path, 'r', 'utf-8') as predicted:

            target_stream = target.read().split()
            predicted_stream = predicted.read().split()

            while True:

                if PUNCTUATION_MAPPING.get(target_stream[t_i], target_stream[t_i]) in PUNCTUATION_VOCABULARY:
                    # skip multiple consecutive punctuations
                    while PUNCTUATION_MAPPING.get(target_stream[t_i], target_stream[t_i]) in PUNCTUATION_VOCABULARY:
                        target_punctuation = PUNCTUATION_MAPPING.get(
                            target_stream[t_i], target_stream[t_i])
                        target_punctuation = MAPPING.get(
                            target_punctuation, target_punctuation)
                        t_i += 1
                else:
                    target_punctuation = " "

                if predicted_stream[p_i] in PUNCTUATION_VOCABULARY:
                    predicted_punctuation = MAPPING.get(
                        predicted_stream[p_i], predicted_stream[p_i])
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
                elif predicted_punctuation != " " and target_punctuation != " " and predicted_punctuation == target_punctuation:
                    correct += 1
                elif predicted_punctuation != " " and target_punctuation != " " and predicted_punctuation != target_punctuation:
                    substitutions += 1

                true_positives[target_punctuation] = true_positives.get(
                    target_punctuation, 0.) + float(is_correct)
                false_positives[predicted_punctuation] = false_positives.get(
                    predicted_punctuation, 0.) + float(not is_correct)
                false_negatives[target_punctuation] = false_negatives.get(
                    target_punctuation, 0.) + float(not is_correct)

                assert target_stream[t_i] == predicted_stream[p_i] or predicted_stream[p_i] == "<unk>", \
                    ("File: %s \n" +
                     "Error: %s (%s) != %s (%s) \n" +
                        "Target context: %s \n" +
                        "Predicted context: %s") % \
                    (target_path,
                     target_stream[t_i], t_i, predicted_stream[p_i], p_i,
                     " ".join(target_stream[t_i-2:t_i+2]),
                     " ".join(predicted_stream[p_i-2:p_i+2]))

                t_i += 1
                p_i += 1

                if t_i >= len(target_stream)-1 and p_i >= len(predicted_stream)-1:
                    break

    overall_tp = 0.0
    overall_fp = 0.0
    overall_fn = 0.0

    print("-"*46)
    print("{:<16} {:<9} {:<9} {:<9}".format(
        'PUNCTUATION', 'PRECISION', 'RECALL', 'F-SCORE'))

    for p in PUNCTUATION_VOCABULARY:

        if p == SPACE:
            continue

        overall_tp += true_positives.get(p, 0.)
        overall_fp += false_positives.get(p, 0.)
        overall_fn += false_negatives.get(p, 0.)

        punctuation = p
        precision = (true_positives.get(p, 0.) / (true_positives.get(p,
                                                                     0.) + false_positives[p])) if p in false_positives else nan
        recall = (true_positives.get(p, 0.) / (true_positives.get(p,
                                                                  0.) + false_negatives[p])) if p in false_negatives else nan
        f_score = (2. * precision * recall / (precision + recall)
                   ) if (precision + recall) > 0 else nan
        print("{:<16} {:<9.1f} {:<9.1f} {:<9.1f}".format(punctuation,
                                                         precision*100, recall*100, f_score*100))
    print("-"*46)
    pre = overall_tp/(overall_tp+overall_fp) if overall_fp else nan
    rec = overall_tp/(overall_tp+overall_fn) if overall_fn else nan
    f1 = (2.*pre*rec)/(pre+rec) if (pre + rec) else nan
    print("{:<16} {:<9.1f} {:<9.1f} {:<9.1f}".format("Overall",
                                                     pre*100, rec*100, f1*100))
    print("Err: {:.2f}".format(
          100.0 - float(total_correct) / float(counter-1) * 100.0))
    print("SER: {:.1f}".format((substitutions + deletions + insertions) /
                               (correct + substitutions + deletions) * 100))


if __name__ == "__main__":

    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        sys.exit("Ground truth file path argument missing")

    if len(sys.argv) > 2:
        predicted_path = sys.argv[2]
    else:
        sys.exit("Model predictions file path argument missing")

    compute_error([target_path], [predicted_path])
