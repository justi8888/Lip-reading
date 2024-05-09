import argparse
import os
from utils import load_files, get_video_name, remove_diacritics, get_sample_rate, create_audio
from google.cloud import speech
from google.oauth2 import service_account




def create_client(cloud_credentials):
    credentials = cloud_credentials
    credentials = service_account.Credentials.from_service_account_file(credentials)
    client = speech.SpeechClient(credentials=credentials)
    return client


def save_transcript_to_txt(string_data, output_dir, file_path):
    file = os.path.join(output_dir, f'{file_path}.txt')
    with open(file, 'w') as f:
        f.write(string_data.upper())
    
    file = os.path.join(f"{output_dir}_nod", f'{file_path}.txt')
    with open(file, 'w') as f:
        removed = remove_diacritics(string_data)
        f.write(removed.upper())
        
        
def transcribe(audio_file, client, hertz_rate, channels, output_dir, video_name):
    # Reads a file as bytes
    
    with open(audio_file, "rb") as f:
        content = f.read()
    audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        # auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        sample_rate_hertz=hertz_rate,
        audio_channel_count = channels,
        language_code='cs-CZ',
        #enable_automatic_punctuation=True,
    )


    # # Transcribes the audio into text
    # response = client.recognize(request=request)
    response = client.recognize(config=config, audio=audio)
    
    
    for i, result in enumerate(response.results):
        alternative = result.alternatives[0]
        save_transcript_to_txt(string_data=alternative.transcript, output_dir=output_dir, file_path=video_name)
    print(f'Audio file {audio_file} processed.')
    return response





def main():
    parser = argparse.ArgumentParser(description="Creating a dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        #required=True,
        default="/data/jkuspalova/snemovna",
        help="Directory in which downloaded videos in subdirectory video are located",
)
    parser.add_argument(
        "--cloud-credentials",
        type=str,
        #required=True,
        default="/data/jkuspalova/sentence_segmentation/transcribing_credentials.json",
        help="Path to JSON credentials",
    )    
    parser.add_argument(
        "--input-type",
        type=str,
        default="mp4",
        help="Extension of the input video. Default: mp4",
    )
    
    args = parser.parse_args()
    input_type = args.input_type
    data_dir = args.data_dir
    video_dir = f"{data_dir}/video_shorts"
    transcript_dir = f"{data_dir}/transcripts"
    audio_dir = f"{data_dir}/audio"
    CLOUD_CREDENTIALS = args.cloud_credentials
    
    
    
    # Create directories for video sequences, audio and transcripts
    if not os.path.exists(audio_dir):
        print(f"Transcript directory {audio_dir} created")
        os.makedirs(audio_dir)
        
    if not os.path.exists(transcript_dir):
        print(f"Transcript directory {transcript_dir} created")
        os.makedirs(transcript_dir)
        
    if not os.path.exists(f'{transcript_dir}_nod'):
        print(f"Transcript directory {transcript_dir}_nod created")
        os.makedirs(f'{transcript_dir}_nod')
        
        
     
    # Load sequences, create Google Cloud client, and transcribe   
    sequences = load_files(dir_path=video_dir, extension="mp4")
    client = create_client(cloud_credentials=CLOUD_CREDENTIALS)

    
    for sequence in sequences:
        video_name = get_video_name(file=sequence)
        audio_file = create_audio(video_file=sequence, output_dir=audio_dir)
        hertz_rate, channels = get_sample_rate(audio_file=audio_file)
        transcribe(audio_file=audio_file, client=client, hertz_rate=hertz_rate, channels=channels, output_dir=transcript_dir, video_name=video_name)
        
        
    try:
        if os.path.exists(audio_dir):
            os.system(f'rm -rf {audio_dir}')
            print(f"Directory '{audio_dir}' and its contents have been successfully deleted.")
    except Exception as e:
        print(f"Error deleting directory '{audio_dir}': {e}")
    
        

if __name__ == "__main__":
    main()