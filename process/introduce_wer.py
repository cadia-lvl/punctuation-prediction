# Copyright 2020 Helga Svala Sigurðardóttir helgas@ru.is
# In this script, the word error rate is introduced to data
# and the data then saved to a file.
from wer_assist import apply_wer
import sys

try:
    wordList_wer = apply_wer(float(sys.argv[3]))
    sentences_wer = [" ".join(sentence) for sentence in wordList_wer]
except:
    print("There is no number to define the desired word error rate")

try:
    with open(sys.argv[2] + "/wer" + sys.argv[3] + ".txt", "w", encoding="utf-8") as show_unurl:
        for item in sentences_wer:
            show_unurl.write("%s\n" % item)
except:
    print("Unable to save to directory")
