# datasets.py

import os
import cv2
import glob
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
import numpy as np

class SingaporeDataset(Dataset):
    """
    Dataset class for real maritime images extracted from .avi videos.
    Frames are extracted and saved to disk to avoid redundant processing.
    """
    def __init__(self, video_dirs, split='train', transform=None,  extraction_dir='singapore/extracted_frames'):
        """
        Args:
            video_dirs (list): List of directories containing .avi video files.
            split (str): 'train' or 'test' split.
            transform (callable, optional): Optional transform to be applied on a sample.
            random_seed (int): Seed for reproducibility.
            extraction_dir (str): Directory to save or load extracted frames.
        """
        self.transform = transform
        self.label = np.array([1.0,0])  # Real images label
        self.split = split
        self.extraction_dir = extraction_dir
        self.name = "singapore"

        os.makedirs(self.extraction_dir, exist_ok=True)


        self.video_dirs = video_dirs
        self.video_paths = []
        for video_dir in self.video_dirs:
            self.video_paths.extend(glob.glob(os.path.join(video_dir, '*.avi')))

        if not self.video_paths:
            raise ValueError(f"No video files found in the provided directories: {video_dirs}")

        self.extract_and_cache_frames()

        self.frame_paths = glob.glob(os.path.join(self.extraction_dir, '**', '*.jpg'), recursive=True)
        if not self.frame_paths:
            raise ValueError(f"No extracted frames found in: {self.extraction_dir}")

        self.frame_paths = sorted(self.frame_paths)

        self.data = self.frame_paths

    def extract_and_cache_frames(self):
        """
        Extract frames from videos and save them to the extraction directory.
        Checks if frames have already been extracted to avoid redundant processing.
        """
        for video_path in self.video_paths:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_extraction_dir = os.path.join(self.extraction_dir, video_name)
            os.makedirs(video_extraction_dir, exist_ok=True)

            existing_frames = glob.glob(os.path.join(video_extraction_dir, '*.jpg'))
            if existing_frames:
                continue 

            print(f"Extracting frames from video")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("Failed")
                continue

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_filename = f"{video_name}_frame{frame_idx:05d}.jpg"
                frame_path = os.path.join(video_extraction_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                frame_idx += 1

            cap.release()
            print(f"Extracted {frame_idx} frames from video: {video_name}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame_path = self.data[idx]
        label = self.label

        # Load image
        image = Image.open(frame_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32), self.name


class ImageDataset(Dataset):
    """
    Dataset class for synthetic maritime images stored as .png files.
    Each image is considered an individual sample with label 0.0.
    """
    def __init__(self, image_dir, label, split='train', transform=None, name = "unnamed"):
        """
        Args:
            image_dir (str): Directory containing .png image files.
            split (str): 'train' or 'test' split.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.label = np.array([0.0,1.0]) if label == "synthetic" else np.array([1.0,0.0]) # Synthetic images label
        self.split = split
        self.name = name

        # Gather all image file paths
        image_extensions = ['.jpg', '.jpeg', '.png']

        image_paths = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    file_path = os.path.join(root, file)
                    if "depth" not in file_path.lower():  # Check if "depth" is not in the path in case 16bit int depth images are also in the dataset
                        image_paths.append(file_path)

        self.image_paths = image_paths
        if not self.image_paths:
            raise ValueError(f"No image files found in: {image_dir}")

        self.image_paths = sorted(self.image_paths)

        self.data = self.image_paths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        label = self.label

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as inst:
            print(inst)  
            idx = 0 #default image in case file is corrupt
            image_path = self.data[idx]
            image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32), self.name