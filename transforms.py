import torch
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms
from scipy.ndimage import sobel

def get_train_transforms(img_size=[640,480], use_gradient=False, use_fourier=False, crop=False):
    if use_gradient:
        return transforms.Compose([
            transforms.RandomCrop((img_size[1], img_size[0])) if crop else transforms.Resize((img_size[1], img_size[0])),  
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5), 
            GradientTransform(),             
        ])
    elif use_fourier:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5), 
            transforms.RandomCrop((img_size[1], img_size[0])) if crop else transforms.Resize((img_size[1], img_size[0])),  
            #transforms.Lambda(lambda img: fourier_transform_magnitude(img, 
            #                                                        log_scale=True,              #Fourier as preprocessing is slow due to processing on the cpu. use gpu/torch to do it vectorized instead
            #                                                        shift=True)),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.RandomCrop((img_size[1], img_size[0])) if crop else transforms.Resize((img_size[1], img_size[0])),  
            transforms.RandomHorizontalFlip(p=0.5),  #flipping is possible as as the signal is independent of orientation
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),          
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
        ])

def get_test_transforms(img_size=[640,480], use_gradient=False, use_fourier=False, crop=False,):
    if use_gradient:
        return transforms.Compose([
            transforms.CenterCrop((img_size[1], img_size[0])) if crop else transforms.Resize((img_size[1], img_size[0])), 
            GradientTransform(),                     # Convert image to its gradient
        ])
    elif use_fourier:
        return transforms.Compose([
            transforms.CenterCrop((img_size[1], img_size[0])) if crop else transforms.Resize((img_size[1], img_size[0])), 
            #transforms.Lambda(lambda img: fourier_transform_magnitude(img, 
            #                                                        log_scale=True, 
            #                                                        shift=True)),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.CenterCrop((img_size[1], img_size[0])) if crop else transforms.Resize((img_size[1], img_size[0])), 
            transforms.ToTensor(),       
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
        ])



class GradientTransform:
    def __call__(self, img):
        """
        Convert the input Image to a single-channel gradient magnitude image that is normalized
        """
        img_gray = np.array(img.convert("L"))  # grayscale
        grad_x = sobel(img_gray, axis=0)  # x-direction
        grad_y = sobel(img_gray, axis=1)  # y-direction
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)  #Compute gradient magnitude
        grad_magnitude = (grad_magnitude - grad_magnitude.min()) / (grad_magnitude.max() - grad_magnitude.min()) 
        return torch.tensor(grad_magnitude, dtype=torch.float32).unsqueeze(0)  