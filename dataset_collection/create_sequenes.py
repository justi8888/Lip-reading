import argparse
import os
from pydub import AudioSegment, silence
from moviepy.editor import VideoFileClip
from utils import load_files, get_video_name, remove_diacritics, get_sample_rate, create_audio

import dlib
import cv2
import math


# Initialize the CUDA-enabled dlib shape predictor
dlib.DLIB_USE_CUDA = True


def identify_sentences(audio_file):
    print(f'Detecting silence in {audio_file}')
    myaudio = AudioSegment.from_wav(audio_file)
    dBFS=myaudio.dBFS
    os.remove(audio_file)
    print(f'Audio {audio_file} removed')
    sil = silence.detect_silence(myaudio, min_silence_len=500, silence_thresh=dBFS-16)
    sil = [((start/1000),(stop/1000)) for start,stop in sil] #in sec
    return sil


def split_video(video_file, timestamps, output_dir, predictor, min_length, max_length, min_lip_distance):
    video_name = get_video_name(file=video_file)
    print(f"Creating video sequences from {video_name}")
    video = VideoFileClip(video_file)
    
    
    previous_end = 0
    for i, (start, end) in enumerate(timestamps):
        # Check the length of video. If there is music in backgroud the clip is too long because there is no silence
        # Also exclude sequences shorter than min_length
        if start-previous_end < max_length and start-previous_end > min_length:
            output_file = os.path.join(output_dir, f"{video_name}_s{i}.mp4")
            if previous_end > 0.1:
                clip = video.subclip(previous_end-0.1, start+0.1)
            else:
                clip = video.subclip(previous_end, start+0.1)
            frames = []
            for frame in clip.iter_frames():
                # Convert the frame to BGR (OpenCV uses BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frames.append(frame)
            
            video_short_name = get_video_name(output_file)
            if detect_landmarks_in_video(video_name=video_short_name, frames=frames, predictor=predictor, min_lip_distance=min_lip_distance):
                clip_25fps = clip.set_fps(25)
                clip_25fps.write_videofile(output_file, codec="libx264", audio_codec="aac")
        previous_end = end
        
    print(f"Video sequences from video {video_name} created")



def is_mouth_open(shape, frame, min_lip_distance=1):
    # Calculate the distance between upper and lower lip
    #print(shape.part(66))
    lip_distance = shape.part(66).y - shape.part(62).y
    return lip_distance > min_lip_distance 


# if mouth was not opened, check next 3 frames if it still is not open
def check_next_three(frames, frame_count, shape, min_lip_distance):
    if len(frames) - frame_count < 4:
        #print(f"LAST FRAMES.")
        return True
    if is_mouth_open(shape=shape, frame=frames[frame_count+1], min_lip_distance=min_lip_distance) or is_mouth_open(shape=shape, frame=frames[frame_count+2], min_lip_distance=min_lip_distance) or is_mouth_open(shape=shape, frame=frames[frame_count+3], min_lip_distance=min_lip_distance):
        #print(f"Checking next two frames after frame {frame_count} successful.")
        return True
    else:
        #print(f"Checking next two frames after frame {frame_count} unsuccessful.")
        return False
    

def check_the_rest(frames, predictor, min_lip_distance=1):
    total_frames = len(frames)
    #print(f"Total frames: {len(frames)}")
    frames_to_skip = int(total_frames / 10)
    frame_count = 0
    for frame in frames:
        if frame_count % frames_to_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = dlib.get_frontal_face_detector()(gray)
            
            # too many faces or no face
            if len(faces) != 1:
                return False
            
            face = faces[0]
            shape = predictor(gray, face)
            
            # check if mouth is open, if not, chcek next three frames
            if not is_mouth_open(shape=shape, frame=frame, min_lip_distance=min_lip_distance):
                if check_next_three(frames=frames, frame_count=frame_count, shape=shape, min_lip_distance=min_lip_distance):
                    continue
                return False 
        frame_count += 1
    #print(f"Video {video_name} contains only speaking.")
    return True
    

def detect_landmarks_in_video(video_name, frames, predictor, min_lip_distance):
    print("------------------------------------------------------")
    print(f"Processing video: {video_name}")
    print("--------------------------------------------------------")
    
    frame_count = 0
    for frame in frames:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the grayscale frame
        faces = dlib.get_frontal_face_detector()(gray)
        if len(faces) > 1:
            #print(f"Too many faces detected in {video_name} frame {frame_count}.")
            return False
        for face in faces:
            # Detect facial landmarks for each face
            shape = predictor(gray, face)
            if is_mouth_open(shape=shape, frame=frame, min_lip_distance=min_lip_distance):
                #print(f"Speaking detected in frame: {frame_count}")
                # if movement is detected check 10 frames proportionally
                return check_the_rest(frames=frames, predictor=predictor, min_lip_distance=min_lip_distance)
        frame_count += 1
        # if speking is not detected in first 3 frames, the video is not suitable
        if frame_count > 3:
            #print(f"Early Speaking not detected in {video_name}")
            return False
    #print(f"Speaking not detected in {video_name}")
    return False



def main():
    parser = argparse.ArgumentParser(description="Creating a dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        #default="/content/drive/MyDrive/reporteri2",
        help="Directory in which downloaded videos are located",
)
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        #default="/content",
        help="Root directory",
)
    parser.add_argument(
        "--path-to-predictor",
        type=str,
        required=True,
        #default="/content/drive/MyDrive/shape_predictor_68_face_landmarks.dat",
        help="Directory of downloaded videos",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=1,
        help="Min length of sequences included in dataset",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=15,
        help="Max length of sequences included in dataset",
    )
    parser.add_argument(
        "--min-lip-distance",
        type=int,
        default=1,
        help="Min distance between lips to include the sequence in dataset",
    )
    parser.add_argument(
        "--groups",
        type=int,
        default = 1,
        #required = True,
        help="Number of"
    )
    parser.add_argument(
        "--worker",
        type=int,
        default = 0,
        #required = True,
        help=""
    )
    
    args = parser.parse_args()
    data_dir = args.data_dir
    root_dir = args.root_dir
    video_dir = f"{root_dir}/video_shorts"
    audio_dir = f"{root_dir}/audio"
    min_length = args.min_length
    max_length = args.max_length
    min_lip_distance = args.min_lip_distance
    groups = args.groups
    worker = args.worker
    predictor = dlib.shape_predictor(args.path_to_predictor)
    


    # Load videos
    filenames = load_files(data_dir, extension="mp4")
    if not os.path.exists(video_dir):
        print(f"Video directory {video_dir} created")
        os.makedirs(video_dir)
        
    if not os.path.exists(audio_dir):
        print(f"Audio directory {audio_dir} created")
        os.makedirs(audio_dir)
    #print(f"total videos: {len(original_videos)}")
    
    if groups > 1:
        unit = math.ceil(len(filenames) * 1.0 / groups)
        #print(f"unit: {unit}")
        filenames = filenames[worker * unit : (worker + 1) * unit]
        #print(filenames)
        

    
    for original_video in filenames:
        audio_file = create_audio(video_file=original_video, output_dir=audio_dir)
        
        # identify silence timestamps and split
        timestamps = identify_sentences(audio_file=audio_file)
        split_video(video_file=original_video, timestamps=timestamps, output_dir=video_dir, predictor=predictor, min_length=min_length, max_length=max_length, min_lip_distance=min_lip_distance)
        
    try:
        if os.path.exists(audio_dir):
            os.system(f'rm -rf {audio_dir}')
            print(f"Directory '{audio_dir}' and its contents have been successfully deleted.")
    except Exception as e:
        print(f"Error deleting directory '{audio_dir}': {e}")
        


if __name__ == "__main__":
    main()