# inference.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models
from datasets import SingaporeDataset, ImageDataset
import os
import random
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import argparse
import pandas as pd
import numpy as np
from transforms import get_test_transforms
from utils import compute_fourier

def downsample_dataset(dataset, fraction=0.2, exclude_indices=None, seed=42):
    """
    Downsample the dataset to the given fraction, ensuring no overlap with exclude_indices.
    """
    random.seed(seed)
    size = int(len(dataset) * fraction)
    all_indices = set(range(len(dataset)))
    if exclude_indices:
        available_indices = list(all_indices - set(exclude_indices))
    else:
        available_indices = list(all_indices)

    if size > len(available_indices):
        raise ValueError("Requested downsample size exceeds available indices after exclusion.")
    
    selected_indices = random.sample(available_indices, size)
    return Subset(dataset, selected_indices), selected_indices

def prepare_test_datasets(test_transform, args):
    real_video_dirs = [
        '{}/VIS_Onshore/VIS_Onshore/Videos'.format(args.singapore_path),
        '{}/VIS_Onboard/VIS_Onboard/Videos'.format(args.singapore_path)
    ]
    extraction_dir = os.path.join(args.singapore_path, "extracted_frames")

    tartanair_exclude = set()
    diode_exclude = set()
    simuships_exclude = set()
    singapore_exclude = set()

    # Define Test Datasets
    tartanair_test = ImageDataset(
        image_dir=args.tartan_path,
        label="synthetic",
        split="test",
        transform=test_transform,
        name="Tartanair"
    )
    tartanair_test, _ = downsample_dataset(tartanair_test, 0.04, exclude_indices=tartanair_exclude)

    aboships = ImageDataset(
        image_dir=args.aboships_path,
        label="real",
        split="train",
        transform=test_transform,
        name="ABOships"
    )
    aboships, _ = downsample_dataset(aboships, 0.1)

    diode_val = ImageDataset(
        image_dir=args.diode_path,
        label="real",
        split="test",
        transform=test_transform,
        name="Diode"
    )
    diode_val, _ = downsample_dataset(diode_val, 0.004, exclude_indices=diode_exclude)

    simuships_val = ImageDataset(
        image_dir=args.simuships_path,
        label="synthetic",
        split="train",
        transform=test_transform,
        name="Simuships"
    )
    simuships_val, _ = downsample_dataset(simuships_val, 0.05, exclude_indices=simuships_exclude)

    singapore_val = SingaporeDataset(
        video_dirs=real_video_dirs,
        split='test',
        transform=test_transform,
        extraction_dir=extraction_dir
    )
    singapore_val, _ = downsample_dataset(singapore_val, 0.1, exclude_indices=singapore_exclude)

    synthia = ImageDataset(
        image_dir=args.synthia_path,
        label="synthetic",
        split="test",
        transform=test_transform,
        name="Synthia"
    )

    nyu = ImageDataset(
        image_dir=args.nyu_path,
        label="real",
        split="test",
        transform=test_transform,
        name="NYU"
    )

    hypersim = ImageDataset(
        image_dir=args.hypersim_path,
        label="synthetic",
        split="test",
        transform=test_transform,
        name="Hypersim"
    )

    # List of all test datasets
    test_datasets = {
        "Tartanair": tartanair_test,
        "ABOships": aboships,
        "Diode": diode_val,
        "Simuships": simuships_val,
        "Singapore": singapore_val,
        "Synthia": synthia,
        "NYU": nyu,
        "Hypersim": hypersim
    }

    return test_datasets

def get_dataloader(dataset, batch_size=32, num_workers=4):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

def define_model(num_classes=2, pretrained=False):
    model = models.resnet18(pretrained=pretrained)

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.Linear(256, num_classes)
    )  

    return model

def evaluate(model, dataloader, device, apply_highpass=False, highpass_cutoff=10):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device).float()

            fourier_images = compute_fourier(
                images,
                log_scale=True,
                shift=True,
                apply_highpass=apply_highpass,
                highpass_cutoff=highpass_cutoff
            )

            outputs = model(fourier_images)
            _, preds = torch.max(outputs, 1)  
            _, labels = torch.max(labels, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds) * 100
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    return accuracy, cm

def main(args):
    # Set random seeds for reproducibility
    torch.manual_seed(1)
    random.seed(1)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transformations
    test_transform = get_test_transforms(
        img_size=args.img_size, 
        use_gradient=args.use_gradient, 
        use_fourier=args.use_fourier, 
        crop=args.crop
    )

    # Prepare test datasets
    test_datasets = prepare_test_datasets(test_transform, args)

    # Define the model
    model = define_model(num_classes=2, pretrained=False)
    model.to(device)

    # Load the trained model
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found at {args.model_path}")
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {args.model_path}")

    # Evaluate each dataset
    results = []
    for dataset_name, dataset in test_datasets.items():
        print(f"Evaluating on {dataset_name} dataset with {len(dataset)} samples.")
        dataloader = get_dataloader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        
        accuracy, cm = evaluate(
            model, dataloader, device, 
            apply_highpass=args.apply_highpass, 
            highpass_cutoff=args.highpass_cutoff
        )
        results.append({
            "Dataset": dataset_name,
            "Accuracy": accuracy,
            "Confusion Matrix": cm
        })
        

    # Calculate total accuracy
    total_correct = sum([cm[i][i] for cm in [res["Confusion Matrix"] for res in results] for i in range(len(cm))])
    total_samples = sum([np.sum(cm) for cm in [res["Confusion Matrix"] for res in results]])
    total_accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0

    # Generate Markdown Report
    markdown = f"# Model Inference Results\n\n"
    markdown += f"**Total Accuracy Across All Datasets:** {total_accuracy:.2f}%\n\n"

    # Per-Dataset Accuracy Table
    markdown += "## Per-Dataset Accuracy\n\n"
    markdown += "| Dataset | Accuracy (%) |\n"
    markdown += "|---------|--------------|\n"
    for res in results:
        markdown += f"| {res['Dataset']} | {res['Accuracy']:.2f} |\n"
    markdown += "\n"

    # Confusion Matrices
    markdown += "## Confusion Matrices\n\n"
    for res in results:
        cm = res["Confusion Matrix"]
        print(cm.shape)
        if cm.shape != (2, 2):
            markdown += f"### {res['Dataset']}\n\n"
            markdown += "Confusion matrix is not 2x2 and cannot be displayed properly.\n\n"
            continue

        markdown += f"### {res['Dataset']}\n\n"
        markdown += "|               | Predicted Synthetic | Predicted Real |\n"
        markdown += "|---------------|---------------------|----------------|\n"
        markdown += f"| **Actual Synthetic** | {cm[1][1]} | {cm[1][0]} |\n"
        markdown += f"| **Actual Real**      | {cm[0][1]} | {cm[0][0]} |\n\n"

    # Print Markdown
    print("## Inference Results (Markdown):\n")
    print("```markdown")
    print(markdown)
    print("```")

if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser(description="Inference script for maritime classifier.")
        parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model .pth file')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
        parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
        parser.add_argument('--use_gradient', action='store_true', help='Use gradient transform')
        parser.add_argument('--use_fourier', action='store_true', help='Use Fourier transform')
        parser.add_argument('--crop', action='store_true', help='Crop instead of resize')
        parser.add_argument('--img_size', type=int, nargs=2, default=[640,480], help="Image size for evaluation")
        parser.add_argument('--apply_highpass', action='store_true', help='Apply high-pass filter during inference')
        parser.add_argument('--highpass_cutoff', type=int, default=10, help='High-pass filter cutoff frequency')

        # Dataset paths
        parser.add_argument('--tartan_path', type=str, default='../tartanair', help='Path to TartanAir dataset')
        parser.add_argument('--diode_path', type=str, default='../diode', help='Path to Diode dataset')
        parser.add_argument('--singapore_path', type=str, default='../singapore', help='Path to Singapore dataset')
        parser.add_argument('--aboships_path', type=str, default='../ABOshipsDataset', help='Path to ABOships dataset')
        parser.add_argument('--simuships_path', type=str, default='../simuships', help='Path to Simuships dataset')
        parser.add_argument('--synthia_path', type=str, default='../synthia', help='Path to Synthia dataset')
        parser.add_argument('--nyu_path', type=str, default='../nyu_test/color', help='Path to NYU dataset')
        parser.add_argument('--hypersim_path', type=str, default='../ml-hypersim/contrib/99991/downloads', help='Path to Hypersim dataset')
        
        return parser.parse_args()

    args = parse_args()
    main(args)
