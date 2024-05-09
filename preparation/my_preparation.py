from transforms import TextTransform
from data.data_module import AVSRDataLoader
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
        default='/data/jkuspalova/snemovna',
        help="Directory of video sequences",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        #required=True,
        default='/data/jkuspalova/my_snemovna',
        help="Root directory of preprocessed dataset",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        #required=True,
        default='my_snemovna',
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
    input_video_dir = os.path.join(data_dir, "video_shorts")
    
    video_loader = AVSRDataLoader(detector="retinaface")
    
    
    label_dir =  os.path.join(root_dir, "labels")
    video_dir = os.path.join(root_dir, DATASET, f"{DATASET}_video")
    text_dir = os.path.join(root_dir, DATASET, f"{DATASET}_text")
    text_nod_dir = os.path.join(root_dir, DATASET, f"{DATASET}_text_nod")
    
    csv_file = os.path.join(label_dir, f"{DATASET}_list.csv")
    char_positions = {char: str(index) for index, char in enumerate(char_lists.char_list)}
    char_positions_nod = {char: str(index) for index, char in enumerate(char_lists.char_list_nod)}


    video_files = load_files(input_video_dir, "mp4")
    random.shuffle(video_files)
    
    
    #if not os.path.exists(os.path.join(root_dir, "labels")):
    os.makedirs(label_dir, exist_ok=True)
        
    #if not os.path.exists(os.path.join(root_dir, DATASET, f"{DATASET}_text_nod")):
    os.makedirs(text_nod_dir, exist_ok=True)
        
    #if not os.path.exists(os.path.join(root_dir, DATASET, f"{DATASET}_text")):
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    
    
    for video_file in video_files:
        video_name = get_video_name(video_file)
        print(f"Processing: {video_name}")
        # expects transcript in transcripts directory
        text_file = os.path.join(data_dir, "transcripts", f"{video_name}.txt")
        if not os.path.exists(text_file):
            print(f"Transcript {text_file} does not exist.")
            continue
        
        with open(text_file, 'r+', encoding='utf-8') as file:
            text_data = file.read()
            # in case that transcript is not in uppercase
            text_data = text_data.upper()
        output_text_path = os.path.join(text_dir, f"{video_name}.txt")
        
        
        if not os.path.exists(os.path.join(video_dir, f"{video_name}.mp4")):
            video_data = video_loader.load_data(video_file)
            output_video_path = os.path.join(video_dir, f"{video_name}.mp4")
            save_vid_txt(output_video_path, output_text_path, video_data, text_data, video_fps=25)
        else:
            print(f"Processing: only text of {video_name}")
            with open(output_text_path, 'w', encoding='utf-8') as f:
                f.write(text_data)
            
            
        text_nod_data = remove_diacritics(text_data)
        output_text_nod_path = os.path.join(text_nod_dir, f"{video_name}.txt")
        with open(output_text_nod_path, 'w') as f:
            f.write(text_nod_data.upper())
        
    
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
