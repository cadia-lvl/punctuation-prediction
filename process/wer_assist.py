# Copyright 2020 Helga Svala Sigurðardóttir helgas@ru.is
# In this file, a helper function to introduce word error 
# rates to a dataset is defined.
# A list of sentences is split up to words, punctuation
# marks removed and a function defined to introduce
# word error rate with 25% deletion, 25% insertion and 
# 50% substitution.
import sys
import random
import numpy as np

# Processed text read from file
if len(sys.argv) > 1:
    if isinstance(sys.argv[1], str):
        with open(sys.argv[1], "r") as file:
            lines = file.read()
        processed_text = lines.split("\n")
else:
    print("There is no specified text to process")

# Split the sentences into words
processed_words = [sentence.split() for sentence in processed_text]

# Make the list flat and find the unique words
flat_words = [item for sublist in processed_words for item in sublist]
unique_words = list(set(flat_words))

print(f"Unique words in text: {len(unique_words)}")

# Remove the words from the text that are just punctuation marks
# if they occur in the text
punctList = [
    ".PERIOD",
    "?QUESTIONMARK",
    "!EXCLAMATIONMARK",
    ",COMMA",
    ";SEMICOLON",
    ":COLON",
    "-DASH",
]

for elem in punctList:
    if elem in unique_words:
        delIdx = unique_words.index(elem)
        del unique_words[delIdx]

# See how often different punctuation marks occur in the text
punctCountsTmp = [flat_words.count(punct) for punct in punctList]
punctCounts = list(zip(punctList, punctCountsTmp))

print(f"The occurrences of punctuation marks in the text: {punctCounts}")

# Apply word error rate, with an input of how much error rate, which data to
# apply it on, and which word list should be used to insert into the list
def apply_wer(
    nperc,
    wordList=processed_words[: int(0.2 * len(processed_words))],
    randomWords=unique_words,
    punctuations=punctList,
):
    scoreList = []
    # Give each word a random score between 0 and 1
    for i in range(len(wordList)):
        scoreList.append(list(np.random.uniform(0, 1, len(wordList[i]))))
    # for score<25% the word is deleted
    # for 25%<=score<50% a new word is inserted after the word in index
    # for 50%<=score the indexed word is substituted for a random word in the word list
    dels = 0.25 * nperc
    ins = dels + 0.25 * nperc
    subs = ins + 0.5 * nperc
    for i in range(len(scoreList)):
        for j in range(len(scoreList[i])):
            # do not do anything to the punctuation marks in the word list
            if wordList[i][j] not in punctuations:
                if scoreList[i][j] < dels:
                    wordList[i][j] = "DELETION"
                elif scoreList[i][j] < ins:
                    wordList[i].insert(j + 1, random.choice(randomWords))
                elif scoreList[i][j] < subs:
                    wordList[i][j] = random.choice(randomWords)

    # Remove the words marked as deleted
    wordListFinish = [
        list(filter(lambda x: x not in ["DELETION"], sublist)) for sublist in wordList
    ]
    return wordListFinish
