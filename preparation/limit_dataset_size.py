import csv
import pandas as pd    
import numpy as np  
import os
import math
        

def limit_size(csv_file, length_in_hours, fps=25):    
    df = pd.read_csv(csv_file)
    target_number_of_frames = length_in_hours * 3600 * fps
    
    #selected_rows = []
    current_sum = 0
    selected_indices = []  # To keep track of selected row indices
    
    while current_sum < target_number_of_frames:
        row_index = np.random.randint(0, len(df))
        if row_index in selected_indices:
            continue
        
        current_sum += df.iloc[row_index, 2]
        selected_indices.add(row_index)
    selected_rows = df.iloc[selected_indices]
    
    filename = os.path.basename(csv_file)
    selected_rows.to_csv(f"{length_in_hours}_{filename}", index=False, header=False)
    return pd.DataFrame(selected_rows)
    
def train_test_split(from_csv, file, )
   
    
    total_length = original_data.shape[0]
    n_val = math.ceil(total_length/100*args.val_size)
    n_test = math.ceil(total_length/100*args.test_size)


    val = original_data.sample(n=n_val, random_state=42)  
    val.to_csv(os.path.join(label_dir, f"val_{DATASET}_list.csv"), index=False)
    original_data.drop(val.index, inplace=True)

    test = original_data.sample(n=n_test, random_state=42)  
    test.to_csv(os.path.join(label_dir, f"test_{DATASET}_list.csv"), index=False)
    original_data.drop(test.index, inplace=True)

    original_data.to_csv(os.path.join(label_dir, f"train_{DATASET}_list.csv"), index=False)
    