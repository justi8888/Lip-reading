#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torchvision


class AVSRDataLoader:
    def __init__(self, detector="retinaface", convert_gray=False):
        if detector == "retinaface":
            from detectors.retinaface.detector import LandmarksDetector
            from detectors.retinaface.video_process import VideoProcess

            self.landmarks_detector = LandmarksDetector(device="cuda:0")
            self.video_process = VideoProcess(convert_gray=convert_gray)

        if detector == "mediapipe":
            from detectors.mediapipe.detector import LandmarksDetector
            from detectors.mediapipe.video_process import VideoProcess

            self.landmarks_detector = LandmarksDetector()
            self.video_process = VideoProcess(convert_gray=convert_gray)

    def load_data(self, data_filename, landmarks=None, transform=True):
        video = self.load_video(data_filename)
        if not landmarks:
            landmarks = self.landmarks_detector(video)
        video = self.video_process(video, landmarks)
        if video is None:
            raise TypeError("video cannot be None")
        video = torch.tensor(video)
        return video



    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()


