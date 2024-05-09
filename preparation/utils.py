import os
import glob
import torchvision
import unicodedata
import wave
import moviepy.editor as mp





def split_file(filename, max_frames=600, fps=25.0):

    lines = open(filename).read().splitlines()

    flag = 0
    stack = []
    res = []

    tmp = 0
    start_timestamp = 0.0

    threshold = max_frames / fps

    for line in lines:
        if "WORD START END ASDSCORE" in line:
            flag = 1
            continue
        if flag:
            word, start, end, score = line.split(" ")
            start, end, score = float(start), float(end), float(score)
            if end < tmp + threshold:
                stack.append(word)
                last_timestamp = end
            else:
                res.append(
                    [
                        " ".join(stack),
                        start_timestamp,
                        last_timestamp,
                        last_timestamp - start_timestamp,
                    ]
                )
                tmp = start
                start_timestamp = start
                stack = [word]
    if stack:
        res.append([" ".join(stack), start_timestamp, end, end - start_timestamp])
    return res


def save_vid_txt(
    dst_vid_filename, dst_txt_filename, trim_video_data, content, video_fps=25
):
    # -- save video
    save2vid(dst_vid_filename, trim_video_data, video_fps)
    # -- save text
    os.makedirs(os.path.dirname(dst_txt_filename), exist_ok=True)
    
    with open(dst_txt_filename, "w", encoding='utf-8') as file:
        file.write(f"{content}")
    # f = open(dst_txt_filename, "w")
    # f.write(f"{content}")
    # f.close()


def save2vid(filename, vid, frames_per_second):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torchvision.io.write_video(filename, vid, frames_per_second)
    

def load_files(folder_path, extension):
    files = glob.glob(os.path.join(folder_path, f"*.{extension}"))
    return files
    

def get_video_name(file):
    video_name = os.path.basename(file)
    video_name = os.path.splitext(video_name)[0]
    return video_name
    
def remove_diacritics(input_string):
    normalized_string = unicodedata.normalize('NFKD', input_string)
    removed_diacritics = ''.join(char for char in normalized_string if not unicodedata.combining(char))
    return removed_diacritics


def get_sample_rate(audio_file):
    with wave.open(audio_file, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        return sample_rate, channels
    
    
def create_audio(video_file, output_dir):
    video = mp.VideoFileClip(video_file)
    audio = video.audio
    video_name = get_video_name(file=video_file)
    audio_file = os.path.join(output_dir, f"{video_name}.wav")
    audio.write_audiofile(f"{audio_file}")
    return audio_file




