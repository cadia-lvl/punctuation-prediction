# Punctuation detection in Icelandic

Support tools for punctuation and boundary detection for ASR output.

As of January 13th 2020, the code is available for putting in punctuations and introducing word error rate.

To run it:

Start with a file with a list of strings, e.g, `text_to_process.txt`:

```
Verð á olíu á Asíumörkuðum lækkaði í nótt eftir tilkynningu Sádi Araba.
Það hlýtur að hafa verið eins blaut tuska í andlitið.
Kuldatölurnar sýna tuttugu og þriggja stiga frost í Reykjavík klukkan sex á jóladag.
Á miðnætti var hefðbundin flugeldasýning.
Klukkan fjögur verður svokölluð Fjallkonuhátíð, garðveisla með ýmsum uppákomum.
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

This writes a file with processed_text, split in the data directory and unsplit in the working directory. Punctuator 2 (https://github.com/ottokart/punctuator2) can now be applied to the files.

To introduce a word error rate in the data, run:
```
mkdir {word error rate directory}
python introduce_wer.py {processed_text} {word error rate directory} {word error rate}
```
The function splits the WER to 25% insertions, 25% deletions and 50% substitutions.

The file in the `wer`-directory can be used as a `.dev`-file in punctuator 2, to run `error_calculator.py` against a pretrained model.


