import os
from moviepy.editor import VideoFileClip
import argparse

import csv

def total_frames(csv_file):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        #header = next(reader)  # Skip the header row
        column_sum = 0
        for row in reader:
            column_sum += int(row[2])  
    return column_sum


def main():
    parser = argparse.ArgumentParser(description="Dataset preprocessing")
    parser.add_argument(
        "--input-csv",
        type=str,
        #required=True,
        default='/data/jkuspalova/my_dataset/labels/025_train_my_dataset_list.csv',
        help="Directory of video sequences",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Frames per second of dataset videos",
    )

    args = parser.parse_args()
    csv_file = args.input_csv
    fps = args.fps
    
    total_frames_length = total_frames(csv_file)
    total_duration_seconds = total_frames_length/fps
    total_duration_hours = total_duration_seconds/3600
    
    print("Total number of frames: ", total_frames_length)
    print("Total duration in seconds: ", total_duration_seconds)
    print("Total duration in hours: ", total_duration_hours)
    

    
    
if __name__ == "__main__":
    main()