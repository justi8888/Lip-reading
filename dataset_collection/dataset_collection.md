## Dataset collection
This part contains instructions for creating a dataset for VSR model.

### Downloading
Videos can be downloaded using [yt-dlp](https://github.com/yt-dlp/yt-dlp) command line audio and video downloader. Currently there is an issue with dowloading videos from CT, to download videos from there you need to change from latest version using command: 
```
yt-dlp --update-to bashonly/yt-dlp@ceskatelevize
```
Having a list of links to videos in text file, you can download the videos using command:
```
yt-dlp -P [path_to_dir] -a [path_to_list] -f [file_version] 
```

where 
- `path_to_dir` is a path to the directory where downloaded videos will be saved
- `path_to_list` is a path to list of video links
- `file_version` is version of downloaded file. There are many options that can be shown using `-F` argument, I used `'hls-main-2176'`
I also used argument `--trim-filenames 10` so the final names of the files are not too long. 

### Selecting suitable sequences
First download shape predictor. I used [dlib 68 landmarks shape predictor](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat).
Then run:
```
python create_sequences.py --data_dir [data_dir] \
                           --root_dir [root_dir] \
                           --path-to-predictor [path_to_predictor] \
                           --min-length [min_length] \ 
                           --max-length [max_length] \
                           --min-lip-distance [min_lip_distance] \ 
                           --groups [groups] \ 
                           --worker [worker]
```

<details open>
<summary><strong> Required arguments </strong></summary>

- `data_dir`: Directory where input videos for are located
- `root_dir`: Directory for created sequences
- `path_to_predictor`: Path to shape predictor

</details>

<details open>
<summary><strong> Optional arguments </strong></summary>

- `min_length`: Min length of sequences included in dataset, default: `1`
- `max_length`: Max length of sequences included in dataset, default: `15`
- `min_lip_distance`: Min distance between lips to include the sequence in dataset, default: `3`
- `groups`: Number of workers for parallel processing. Default: `1`, no parallel processing. 
- `worker`: ID of current worker, max value: `groups-1`

</details>

Around 5-10% of videos (depending on the source video type) will suffer from some type of error and won't be suitable. I mention these error cases in my thesis. It is fine to briefly check the output, however it is not necessary to check every single video. 

### Transcribing
I used automatic ASR [Google Cloud Speech-to-Text AI](https://cloud.google.com/speech-to-text?hl=en) to transcribe the video sequences. To use this tool, first an Service Account needs to be created to get JSON key credentials. 

```
python transcribe.py --data-dir [data_dir] --cloud-credentials [cloud_credentials] 
```

- `data_dir` - directory containing subdirectory video_shorts. There are downloaded videos. These videos will not be changed using this step. 
- `cloud_credentials` - path to JSON credentials. 
- `input_type` - extension of video, optional argument, default: `mp4`. 

This script creates 2 transcript directories: `data_dir/transcripts` and `data_dir/transcripts_nod` containing transcripts in upper case, with or without diacritics respectively. During the process directory `data_dir/audio` is created, but it is removed at the end. 


After running this script the dataset is in a structure ready for pre-processing in [preparation](./preparation). 


