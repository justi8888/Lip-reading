import csv
import pandas as pd    
import numpy as np  
import os
import math
from dataset_info import total_frames
        

def limit_size(csv_file, length_in_hours, fps=25):    
    df = pd.read_csv(csv_file)
    target_number_of_frames = length_in_hours * 3600 * fps
    
    #selected_rows = []
    current_sum = 0
    total_sum = df.iloc[:, 2].sum()
    
    if target_number_of_frames > total_sum:
        print("Target length is larger than dataset length. Returning input csv file.")
        return csv_file
    selected_indices = []  # To keep track of selected row indices
    
    while current_sum < target_number_of_frames:
        row_index = np.random.randint(0, len(df))
        if row_index in selected_indices:
            continue
        
        current_sum += df.iloc[row_index, 2]
        selected_indices.append(row_index)
    selected_rows = df.iloc[selected_indices]
    
    
    directory = os.path.dirname(csv_file)
    filename = os.path.basename(csv_file)
    output_filename = os.path.join(directory, f"{length_in_hours}_{filename}")
    selected_rows.to_csv(output_filename, index=False, header=False)
    print(f"CSV file {output_filename} with dataset of length {length_in_hours} created. ")
    return output_filename
    
    
def train_test_split(csv_file, label_dir, val_size, test_size):
    original_data = pd.read_csv(csv_file)
        
    total_length = original_data.shape[0]
    n_val = math.ceil(total_length/100*val_size)
    n_test = math.ceil(total_length/100*test_size)
    
    filename = os.path.basename(csv_file)

    val = original_data.sample(n=n_val, random_state=42)  
    val.to_csv(os.path.join(label_dir, f"val_list.csv"), index=False)
    original_data.drop(val.index, inplace=True)

    test = original_data.sample(n=n_test, random_state=42)  
    test.to_csv(os.path.join(label_dir, f"test_list.csv"), index=False)
    original_data.drop(test.index, inplace=True)

    original_data.to_csv(os.path.join(label_dir, f"train_list.csv"), index=False)
    
    print(f"Train train_{filename}\nvalidation val_{filename}\ntest test_{filename}\nfiles created in {label_dir}.")
    
    
def main():
    csv_file = "/data/jkuspalova/my_dataset/labels/no_nums/my_dataset_list.csv"
    label_dir = "/data/jkuspalova/my_dataset/labels/no_nums"
    
    fps = 25
    total_frames_length = total_frames(csv_file)
    total_duration_seconds = total_frames_length/fps
    total_duration_hours = total_duration_seconds/3600
    
    #limited_csv = limit_size(csv_file=csv_file, length_in_hours=total_duration_hours*0.5)
    train_test_split(csv_file=csv_file, label_dir=label_dir, val_size=5, test_size=5)
    
    
if __name__ == "__main__":
    main()
