from transforms import TextTransform
from utils import save_vid_txt, load_files, get_video_name, remove_diacritics
import os

import csv
import cv2
import argparse
import random
import char_lists
from train_test_split import limit_size, train_test_split


# Replace characters in the input string with their positions
def replace_with_position(input_string, char_positions):
    replaced_chars = [char_positions.get(char, '1') for char in input_string]
    replaced_string = ' '.join(replaced_chars)
    return replaced_string


def append_to_csv(file_path, data):
    with open(file_path, 'a', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["dataset", "rel_path", "input_length", "char_id", "char_nod_id", "token_id"])
        writer.writerow(data)
        

def main():
    parser = argparse.ArgumentParser(description="Dataset preprocessing")
    parser.add_argument(
        "--data-dir",
        type=str,
        #required=True,
        default='/data/jkuspalova/my_dataset/my_dataset',
        help="Directory of video sequences",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        #required=True,
        default='/data/jkuspalova/my_dataset',
        help="Root directory of preprocessed dataset",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        #required=True,
        default='my_dataset',
        help="Name of dataset",
    )
    parser.add_argument(
        "--target-length",
        type=float,
        default=999,
        help="Target length of dataset in hours. Default: full size of dataset."
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=5,
        help="Number of files for testing in percent, integer.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=5,
        help="Number of files for validation in percent, integer.",
    )

    args = parser.parse_args()
    data_dir = args.data_dir
    root_dir = args.root_dir
    DATASET = args.dataset
    target_length = args.target_length
    text_transform = TextTransform()
    #input_video_dir = os.path.join(data_dir, "my_dataset_video")
    
    #label_dir =  os.path.join(root_dir, "labels")
    input_video_dir = '/data/jkuspalova/my_dataset/my_dataset/my_dataset_video'
    label_dir = "/data/jkuspalova/my_dataset/labels"
    csv_file = os.path.join(label_dir, f"ovce_zpravy_list.csv" )
    
    
    char_positions = {char: str(index) for index, char in enumerate(char_lists.char_list)}
    char_positions_nod = {char: str(index) for index, char in enumerate(char_lists.char_list_nod)}


    video_files = load_files(input_video_dir, "mp4")
    print(video_files)
    random.shuffle(video_files)
    #if not os.path.exists(os.path.join(root_dir, "labels")):
    os.makedirs(label_dir, exist_ok=True)
        
    
    for video_file in video_files:
        video_name = get_video_name(video_file)
        if video_name.startswith("o_") or video_name.startswith("z_"):
            print(f"Processing: {video_name}")
            # expects transcript in transcripts directory
            text_file = os.path.join(data_dir, "my_dataset_text", f"{video_name}.txt")
            if not os.path.exists(text_file):
                print(f"Transcript {text_file} does not exist.")
                continue
            
            with open(text_file, 'r+', encoding='utf-8') as file:
                text_data = file.read()
                # in case that transcript is not in uppercase
                text_data = text_data.upper()
                
                
            text_nod_data = remove_diacritics(text_data)
            cap = cv2.VideoCapture(video_file)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            token_list = text_transform.tokenize(text_data)
            token_list = token_list.tolist()
            string_token_list = [str(val) for val in token_list]
            token_id = " ".join(string_token_list)
            
            char_id = replace_with_position(text_data, char_positions)
            char_nod_id = replace_with_position(text_nod_data, char_positions_nod)
            
            
            data_point = {"dataset": f"{DATASET}", "rel_path": f"{DATASET}_video/{video_name}.mp4", "input_length": total_frames, "char_id": char_id, "char_nod_id": char_nod_id, "token_id": token_id}
            append_to_csv(csv_file, data_point)
            
            
    limited_csv = limit_size(csv_file=csv_file, length_in_hours=target_length)
    train_test_split(csv_file=limited_csv, label_dir=label_dir, val_size=5, test_size=5)
    

if __name__ == "__main__":
    main()
