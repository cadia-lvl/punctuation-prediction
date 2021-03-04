## Data

After processing the text (./process), and having files with `.train.txt`, `.dev.txt` and `.test.txt`, the next step is to create the vocabulary.
In `data.py`, define the path in line 13 as where you want your processed data to go. *This path is important when running all files.*
That folder consists of `dev`, `test`, `train`, `vocabulary` and `punctuations`.

In lines 31-32, choose which punctuation marks you want your system to learn. We went with periods, commas and question marks.

Then run:
``` python data.py <data_dir> ```

where `<data_dir>` is the directory of the preprocessed and cleaned `*.train.txt`, `*.dev.txt` and `*.test.txt` files.

## Main

The main file takes in the path to the processed data in `data.py`, `<datadir>` and the model which is defined in `models.py`.
The main and punctuator files might take a long time to run, depending on the size of the data, so it's a good idea to use a GPU.

To run the main-file, type:
``` python main.py <model_name> <hidden_layer_size> <learning_rate> ``` 

`<model_name>`is the name of the model you wish to have in the Model name file.
It's good to put hidden_layer_size = 256 and learning_rate = 0.02.

## Punctuator

To punctuate the data, using the trained model, the command is:

``` python punctuator.py <model_path> <data>.test.txt <model_output_path> ```

`<data>.test.txt` is the file in the data folder which contains a portion of the data in the format:

```mr president ,COMMA ladies and gentlemen ,COMMA in the past ,COMMA since <NUM> ,COMMA in fact ,COMMA un forces have played an important role in stabilising the balkans .PERIOD 
like all the members who have just spoken ,COMMA the commission therefore sincerely regrets the failure to extend the mandate of un troops in the former yugoslav republic of macedonia .PERIOD 
following the recognition of taiwan by the fyrom ,COMMA china decided ,COMMA as you know ,COMMA to exercise its veto in the security council against the extension of the mandate of unpredep .PERIOD 
the presidency of the european union tried to get the authorities in peking and skopje to reach a consensus ,COMMA but these attempts were unsuccessful .PERIOD
```
Note that the .dev file was used to validate the training, use the .test file to punctuate.

`<model_path>`is the path to the trained model `"Model_{model_name}_h{num_hidden}_lr{learning_rate}.pcl"`

`<model_output_path>` is the path to the .txt-file with generated data with punctuation restoration. 
That is, the punctuated version of <data>.dev.txt the model predicts.

## Error calculator

To calculate the precision, recall and F1-score, the output file that your model generated reaches, type:

`python error_calculator.py <data>.test.txt <model_output_path>`

## Play with model

To play with the model, run:
`python play_with_model.py <model_path>`

