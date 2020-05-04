# Punctuation detection in Icelandic

Support tools for punctuation and boundary detection for ASR output.

As of January 13th 2020, the code is available for putting in punctuations and introducing word error rate.

To run it:

Start with a file with a lower-cased (and preferably, for better results, somewhat normalized) list of strings, e.g, `text_to_process.txt`:

```
like all the members who have just spoken, the commission therefore sincerely regrets the failure to extend the mandate of un troops in the former yugoslav republic of macedonia. 
following the recognition of taiwan by the fyrom, china decided, as you know, to exercise its veto in the security council against the extension of the mandate of unpredep. 
the presidency of the european union tried to get the authorities in peking and skopje to reach a consensus, but these attempts were unsuccessful. 
```

Run:
``` 
mkdir {data directory}
python write_to_file.py {text file to process} {data directory} {train file split} {test file split}
```

The default is 80% train, 10% test and 10% evaluation (50% split between the 20% remains of the data):
```
python write_to_file.py text_to_process.txt datadir 0.2 0.5
```

This writes a file with processed_text, split in the data directory and unsplit in the working directory. In ../punctuator2tf2, a tensorflow 2 model using Punctuator 2 (https://github.com/ottokart/punctuator2) can be used to create a Bidirectional RNN model with Attention to the data.

To introduce a word error rate in the data, run:
```
mkdir {word error rate directory}
python introduce_wer.py {processed_text} {word error rate directory} {word error rate}
```
The function splits the WER to 25% insertions, 25% deletions and 50% substitutions.

The file in the `wer`-directory can be used as a `.dev`-file in punctuator 2, to run `error_calculator.py` against a pretrained model.


