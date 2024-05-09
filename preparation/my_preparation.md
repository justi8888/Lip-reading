
# Creating a pre-processed dataset

The model requires videos, transcripts and data lists to be in certain directory structure. This script pre-processes input videos and creates required directory structure. The ROI extraction is done using [Auto AVSR](https://github.com/mpc001/auto_avsr/tree/main/preparation) pipeline. 

```
python my_preparation.py \
--data-dir=[data_dir] \
--root-dir=[root_dir] \
--dataset=[dataset] \
--target-length=[target-length] \
--test_size=[test_size] \ 
--val-size=[val_size]
```

<details open>
<summary><strong> Required arguments </strong></summary>

- `data_dir`: Directory where subdirectory `data_dir/video_shorts` with input videos for pre-processing is located
- `root_dir`: Root directory for pre-processed dataset
- `dataset`: Name of the dataset

</details>

<details open>
<summary><strong> Optional arguments </strong></summary>

- `target_length`: Total target length of dataset in hours, `float`. Default: `full size of dataset`. The whole dataset will be preprocessed first, then CSV file containing random videos whose total length equals target length is created.
- `test_size`: Percentage of test data out of total input data, `integer`. Default: `5`.
- `val_size`: Percentage of validation data out of total input data, `integer`. Default: `5`.

</details>



## Input structure
This script expects video sequences of length 1-15 seconds with a clearly visible speaking face. These files can be obtained with the help of [dataset collection](../dataset_collection). Collected sequences have to be located in the `data_dir` and corresponding transcripts (text files with same names) in directory `data_dir/transcripts`.

```
data_dir
├── input1.mp4
├── input2.mp4
└── transcripts
    ├── input1.txt
    └── input2.txt

```

## Output structure
The output directory structure of pre-processed data is as follows:

```
root_dir
├── labels
│   ├── my_dataset_list.csv
│   ├── test_my_dataset_list.csv
│   ├── train_my_dataset_list.csv
│   └── val_my_dataset_list.csv
└── my_dataset
    ├── my_dataset_text
    │   ├── processed_input1.txt
    │   └── processed_input2.txt
    ├── my_dataset_text_nod
    │   ├── processed_input1.txt
    │   └── processed_input2.txt
    └── my_dataset_video
        ├── processed_input1.txt
        └── processed_input2.txt
```

Filenames `processed_input1` and `processed_input2` are only illustrative, the names of the files remain the same as the names of original input videos e.g. `input1` and `input2` in this case. 


## Structure if CSV files
The structure of CSV file lists `my_dataset_list.csv`, `test_my_dataset_list.csv`, `train_my_dataset_list.csv` and `val_my_dataset_list.csv` is presented below:

```
dataset, rel_path, length, [char_id], [char_nod_id], [token_id]
```
where 
- `dataset` is the name of the dataset
- `rel_path` is the relative path to the video file within the dataset
- `length` is the number of frames in video
- `char_id` is the list of character IDs using diacritics, splitted by whitespace (no comma). Each character in transcript is encoded to a number.
- `char_nod_id` is the list of character IDs without diacritics, using only English letters splitted by whitespace (no comma). Each character in transcript is encoded to a number.
- `token_id` is the list of token IDs tokenized by [SentencePiece model](https://github.com/google/sentencepiece) trained on input text. To trancribe text into tokens use[TextTransform.tokenize](./preparation/transforms.py) method provided by [Auto AVSR](https://github.com/mpc001/auto_avsr). Tokens are splitted by whitespace (no comma).


