## Other functions

### Length of dataset
File `dataset_info.py` computes total length of a data list in CSV file. Prints length in frames, seconds and hours. 

```
python dataset_info.py \
--input_csv=[input_csv] \
--fps=[fps] 
```

where 
- `input_csv` is path to the CSV data list
- `fps` is the frame per second rate of videos in the dataset. `Default`: 25. 