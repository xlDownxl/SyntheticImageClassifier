import torch
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms

def get_train_transforms(img_size=[640,480], crop=False):
    return transforms.Compose([
        transforms.RandomCrop((img_size[1], img_size[0])) if crop else transforms.Resize((img_size[1], img_size[0])),  
        transforms.RandomHorizontalFlip(p=0.5),  #flipping is possible as as the signal is independent of orientation
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),          
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
    ])

def get_test_transforms(img_size=[640,480], crop=False,):
    return transforms.Compose([
        transforms.CenterCrop((img_size[1], img_size[0])) if crop else transforms.Resize((img_size[1], img_size[0])), 
        transforms.ToTensor(),       
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
    ])
