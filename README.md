# punctuation-detection

Support tools for punctuation and boundary detection for ASR output.

As of January 13th 2020, the code is available for putting in punctuations and introducing word error rate.

To run it:

Start with a file with a list of strings, e.g, text_to_process.txt:

```Verð á olíu á Asíumörkuðum lækkaði í nótt eftir tilkynningu Sádi Araba.
Það hlýtur að hafa verið eins blaut tuska í andlitið.
Kuldatölurnar sýna tuttugu og þriggja stiga frost í Reykjavík klukkan sex á jóladag.
Á miðnætti var hefðbundin flugeldasýning.
Klukkan fjögur verður svokölluð Fjallkonuhátíð, garðveisla með ýmsum uppákomum.
```
Run:
``` mkdir datadir workdir
python {text file to process} {train file split} {test file split}
```

The default is 80% train, 10% test and 10% evaluation (50% split between the 20% remains of the data):
```python test_to_process.txt 0.2 0.5
```

This writes a file with processed_text, split in the data directory



