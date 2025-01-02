# main.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import models
from datasets import SingaporeDataset, ImageDataset
import os
import random
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from transforms import *
import argparse
from torchvision.utils import save_image
import pandas as pd
import datetime
from utils import *


def downsample_dataset(dataset, fraction=0.2, exclude_indices=None):
    """
    Downsample the dataset to the given fraction, ensuring no overlap with exclude_indices.
    """
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


def prepare_datasets(train_transform, test_transform, args):
    singapore_video_dirs = [
        f"{args.singapore_path}/VIS_Onshore/VIS_Onshore/Videos",
        f"{args.singapore_path}/VIS_Onboard/VIS_Onboard/Videos"
    ]
    extraction_dir = os.path.join(args.singapore_path, "extracted_frames")


    # Train Data
    tartanair_exclude = set()
    diode_exclude = set()
    singapore_exclude = set()
    simuships_exclude = set()

    tartanair = ImageDataset(
        image_dir=args.tartan_path,
        label="synthetic",
        split="train",
        transform=train_transform,
        name="tartan"
    )
    tartanair, tartanair_exclude = downsample_dataset(tartanair, 0.053)

    simuships = ImageDataset(
        image_dir=args.simuships_path,
        label="synthetic",
        split="train",
        transform=train_transform,
        name="simuships"
    )
    simuships, simuships_exclude = downsample_dataset(simuships, 0.95)

    singapore = SingaporeDataset(
        video_dirs=singapore_video_dirs,
        split='test',
        transform=train_transform,
        extraction_dir=extraction_dir
    )
    singapore, singapore_exclude = downsample_dataset(singapore, 0.4)

    diode = ImageDataset(
        image_dir=args.diode_path,
        label="real",
        split="train",
        transform=train_transform,
        name="diode"
    )
    diode, diode_exclude = downsample_dataset(diode, 0.2)

    # Test Data
    tartanair_test = ImageDataset(
        image_dir=args.tartan_path,
        label="synthetic",
        split="test",
        transform=test_transform,
        name="tartan"
    )
    tartanair_test, _ = downsample_dataset(tartanair_test, 0.04, exclude_indices=tartanair_exclude)

    aboships = ImageDataset(
        image_dir=args.aboships_path,
        label="real",
        split="train",
        transform=test_transform,
        name="aboships"
    )
    aboships, _ = downsample_dataset(aboships, 0.1)

    diode_val = ImageDataset(
        image_dir=args.diode_path,
        label="real",
        split="test",
        transform=test_transform,
        name="diode"
    )
    diode_val, _ = downsample_dataset(diode_val, 0.004, exclude_indices=diode_exclude)

    simuships_val = downsample_dataset(
        ImageDataset(
            image_dir=args.simuships_path,
            label="synthetic",
            split="train",
            transform=test_transform,
            name="simuships"
        ),
        0.05,
        exclude_indices=simuships_exclude
    )[0]

    singapore_val = downsample_dataset(
        SingaporeDataset(
            video_dirs=singapore_video_dirs,
            split='test',
            transform=test_transform,
            extraction_dir=extraction_dir
        ),
        0.1,
        exclude_indices=singapore_exclude
    )[0]

    synthia = ImageDataset(
        image_dir=args.synthia_path,
        label="synthetic",
        split="test",
        transform=test_transform,
        name="synthia"
    )
    nyu = ImageDataset(
        image_dir=args.nyu_path,
        label="real",
        split="test",
        transform=test_transform,
        name="nyu"
    )
    hypersim = ImageDataset(
        image_dir=args.hypersim_path,
        label="synthetic",
        split="test",
        transform=test_transform,
        name="hypersim"
    )

    print("Training dataset balance:")
    print(f"Tartanair Set: {len(tartanair)} images")
    print(f"Diode Set: {len(diode)} images")
    print(f"Simuships Set: {len(simuships)} images")
    print(f"Singapore Set: {len(singapore)} images")

    train_dataset = ConcatDataset([simuships, singapore, tartanair, diode])
    val_dataset = ConcatDataset([
        tartanair_test,
        aboships,
        diode_val,
        simuships_val,
        singapore_val,
        synthia,
        nyu,
        hypersim
    ])

    print(f"Training Set: {len(train_dataset)} images")
    print(f"Validation Set: {len(val_dataset)} images")
    
    return train_dataset, val_dataset


def get_dataloaders(train_dataset, val_dataset, batch_size=32, num_workers=4):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader


def define_model(num_classes=2, pretrained=True):
    model = models.resnet18(pretrained=pretrained)

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.Linear(256, num_classes)
    )  

    return model


def train_model(model, train_loader, val_loader, criterion, optimizer, device, output_dir, save_path, num_epochs=10, use_fourier= True, apply_highpass=False, highpass_cutoff=10, warmup_epochs=1, warmup_lr=0.00001, ):
    model.to(device)
    model.train()
    evaluate_model(
        model,
        val_loader,
        device,
        output_dir=output_dir,
        epoch=-1,
        iteration=-1,
        use_fourier = use_fourier,
        apply_highpass=apply_highpass,
        highpass_cutoff=highpass_cutoff
    )

    original_lr = optimizer.param_groups[0]['lr']
    if warmup_epochs > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        counter = 0
        for images, labels, name in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            labels = labels.to(device).float()

            # Zero the parameter gradients
            optimizer.zero_grad()

            if use_fourier:
                with torch.no_grad():
                    fourier_images = compute_fourier(images, log_scale=True, shift=True, apply_highpass=apply_highpass, highpass_cutoff=highpass_cutoff)

                # Forward pass
                outputs = model(fourier_images)
            else:
                # Forward pass
                outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            counter += 1

        if epoch == -1:  # Restore learning rate after warm-up
            for param_group in optimizer.param_groups:
                param_group['lr'] = original_lr

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        evaluate_model(
            model,
            val_loader,
            device,
            output_dir=output_dir,
            epoch=epoch,
            iteration=counter,
            use_fourier = use_fourier,
            apply_highpass=apply_highpass, 
            highpass_cutoff=highpass_cutoff
        )
        if epoch >= 0:
            model_save_path = os.path.join(output_dir, f"{save_path}_{epoch}.pth")
            torch.save(model.state_dict(), model_save_path)

    return model


def evaluate_model(model, val_loader, device, output_dir='evaluation_outputs', samples_per_class=15, epoch=None, iteration=None, visualize=True, use_fourier = True, apply_highpass=False, highpass_cutoff=10):
    model.to(device)
    model.eval()

    all_preds = torch.tensor([], dtype=torch.long, device=device)
    all_labels = torch.tensor([], dtype=torch.long, device=device)
    all_names = [] 

    correct_samples = {'Synthetic': [], 'Real': []}
    misclassified_samples = {'Synthetic_as_Real': [], 'Real_as_Synthetic': []}
    max_correct_samples = samples_per_class
    max_misclassified_samples = 10

    with torch.no_grad():
        for images, labels, names in val_loader:
            images = images.to(device)
            labels = labels.to(device).float()  
            if use_fourier:
                fourier_images = compute_fourier(images, log_scale=True, shift=True, apply_highpass=apply_highpass, highpass_cutoff=highpass_cutoff)

                # Forward pass
                outputs = model(fourier_images)
            else:
                outputs = model(images)

            _, preds = torch.max(outputs, 1)  
            _, labels = torch.max(labels, 1)

            all_preds = torch.cat([all_preds, preds])
            all_labels = torch.cat([all_labels, labels])
            all_names.extend(names) 

            if visualize:
                for img, true_label, pred_label, name in zip(images, labels, preds, names):
                    true_class = 'Synthetic' if true_label.item() == 1 else 'Real'
                    pred_class = 'Synthetic' if pred_label.item() == 1 else 'Real'

                    #Collect correctly classified samples
                    if true_label.item() == pred_label.item() and len(correct_samples[true_class]) < max_correct_samples:
                        img_np = img.cpu().numpy().transpose((1, 2, 0))
                        img_np = np.clip(img_np, 0, 1)
                        
                        correct_samples[true_class].append({
                            'image': img_np,
                            'true_label': true_class,
                            'pred_label': pred_class,
                            "name": name,
                        })
                    
                    #Collect misclassified samples
                    elif true_label.item() != pred_label.item():
                        if true_class == 'Synthetic' and len(misclassified_samples['Synthetic_as_Real']) < max_misclassified_samples:
                            img_np = img.cpu().numpy().transpose((1, 2, 0))
                            img_np = np.clip(img_np, 0, 1)
                            
                            misclassified_samples['Synthetic_as_Real'].append({
                                'image': img_np,
                                'true_label': true_class,
                                'pred_label': pred_class,
                                "name": name
                            })
                        elif true_class == 'Real' and len(misclassified_samples['Real_as_Synthetic']) < max_misclassified_samples:
                            img_np = img.cpu().numpy().transpose((1, 2, 0))
                            img_np = np.clip(img_np, 0, 1)
                            
                            misclassified_samples['Real_as_Synthetic'].append({
                                'image': img_np,
                                'true_label': true_class,
                                'pred_label': pred_class,
                                "name": name
                            })
 
                    if (len(correct_samples['Synthetic']) >= max_correct_samples and
                        len(correct_samples['Real']) >= max_correct_samples and
                        len(misclassified_samples['Synthetic_as_Real']) >= max_misclassified_samples and
                        len(misclassified_samples['Real_as_Synthetic']) >= max_misclassified_samples):
                        break  #Exit early if all samples are collected

    all_preds = all_preds.cpu().numpy()
    all_labels = all_labels.cpu().numpy()

    df = pd.DataFrame({
        'Dataset': all_names,
        'True Label': all_labels,
        'Predicted Label': all_preds
    })

    per_dataset_accuracy = df.groupby('Dataset').apply(lambda x: accuracy_score(x['True Label'], x['Predicted Label'])).reset_index()
    per_dataset_accuracy.columns = ['Dataset', 'Accuracy']

    per_dataset_accuracy['Accuracy'] = per_dataset_accuracy['Accuracy'] * 100

    #Sort by accuracy descending
    per_dataset_accuracy = per_dataset_accuracy.sort_values(by='Accuracy', ascending=False)

    print("Per-Dataset Accuracy:")
    print(per_dataset_accuracy.to_string(index=False))

    # Calculate overall accuracy
    mean_accuracy = per_dataset_accuracy['Accuracy'].mean()
    print(f"Mean Accuracy Across Datasets: {mean_accuracy:.2f}%")

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Function to plot and save samples
    def plot_and_save(sample_dict, title, filename_prefix):
        for key, sample_list in sample_dict.items():
            if not sample_list:
                print(f"No samples collected for: {key}")
                continue
            
            num_samples = len(sample_list)
            cols = min(num_samples, 10)  # Limit to 10 columns for readability
            rows = 1
            fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3))
            fig.suptitle(title, fontsize=16)
            
            # If only one sample, axes might not be iterable
            if num_samples == 1:
                axes = [axes]
            
            for idx, sample in enumerate(sample_list[:cols]):
                ax = axes[idx]
                ax.imshow(sample['image'])
                ax.axis('off')
                ax.set_title(f"True: {sample['true_label']}\nPred: {sample['pred_label']}\nName: {sample['name']}")
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_path = os.path.join(output_dir, f'epoch_{epoch}_iteration_{iteration}_{filename_prefix}_{key}.png')
            plt.savefig(save_path)
            plt.close()
            print(f"Saved {filename_prefix} samples to {save_path}")

    if samples_per_class > 0:
        plot_and_save(correct_samples, 'Correctly Classified Samples', 'correct_sample_predictions')

    plot_and_save(misclassified_samples, 'Misclassified Samples', 'misclassified_samples')

    return mean_accuracy, cm


def main(args):
    torch.manual_seed(1)
    random.seed(1)
    eval_only = args.eval_only
    model_path = args.model_path
    compute_mean_std = False

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    slurm_job_name = os.environ.get('SLURM_JOB_NAME')
    slurm_job_id = os.environ.get('SLURM_JOB_ID')

    folder_name = current_time
    if slurm_job_name and slurm_job_id:
        folder_name += f"_{slurm_job_name}_{slurm_job_id}"
    elif slurm_job_name:
        folder_name += f"_{slurm_job_name}"
    elif slurm_job_id:
        folder_name += f"_{slurm_job_id}"

    base_output_dir = "logs"
    output_dir = os.path.join(base_output_dir, folder_name)

    os.makedirs(output_dir, exist_ok=True)

    # Define transformations
    train_transform = get_train_transforms(
        img_size=args.img_size, 
        use_gradient=args.use_gradient, 
        use_fourier=args.use_fourier, 
        crop=args.crop
    )
    test_transform = get_test_transforms(
        img_size=args.img_size, 
        use_gradient=args.use_gradient, 
        use_fourier=args.use_fourier, 
        crop=args.crop
    )

    #Prepare and balance datasets
    train_dataset, val_dataset = prepare_datasets(
        train_transform=train_transform, 
        test_transform=test_transform,
        args=args,
    )

    #Create DataLoaders
    train_loader, val_loader = get_dataloaders(
        train_dataset, val_dataset, 
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    if compute_mean_std:
        compute_image_statistics(train_loader, apply_highpass=args.apply_highpass, highpass_cutoff=args.highpass_cutoff)


    model = define_model(num_classes=2, pretrained=args.pretrained) 

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if eval_only:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path} for evaluation")
    else:
        num_epochs = args.num_epochs
        model = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            output_dir=output_dir,
            save_path=args.save_path,
            num_epochs=num_epochs,
            use_fourier= args.use_fourier,
            apply_highpass=args.apply_highpass,
            highpass_cutoff=args.highpass_cutoff,
            warmup_epochs=args.warmup_epochs, 
            warmup_lr=args.warmup_lr,
        )


if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser(description="Train and evaluate a maritime classifier.")
        parser.add_argument('--eval_only', action='store_true', help='Only evaluate the model without training')
        parser.add_argument('--model_path', type=str, default='resnet18_synthetic_classifier', help='Path to the trained model')
        parser.add_argument('--save_path', type=str, default='resnet18_synthetic_classifier', help='Path to save the trained model')
        parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation')
        parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for data loading')
        parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
        
        parser.add_argument('--crop', action='store_true', help='Crop instead of resize')
        parser.add_argument('--img_size', type=int, nargs=2, default=[640,480], help="Image size for training and evaluation")
        parser.add_argument('--pretrained', action='store_true', help="Use pretrained weights for the model")

        parser.add_argument('--use_gradient', action='store_true', help='Use gradient transform')
        parser.add_argument('--use_fourier', action='store_true', help='Use Fourier transform')
        parser.add_argument('--apply_highpass', action='store_true', help='Apply high-pass filter')
        parser.add_argument('--highpass_cutoff', type=int, default=10, help='High-pass filter cutoff frequency')
        parser.add_argument('--warmup_epochs', type=int, default=1, help='Number of warm-up epochs')
        parser.add_argument('--warmup_lr', type=float, default=0.00001, help='Learning rate during warm-up')

        parser.add_argument('--tartan_path', type=str, default='../tartanair', help='Path to TartanAir dataset')
        parser.add_argument('--diode_path', type=str, default='../diode', help='Path to Diode dataset')
        parser.add_argument('--tum_path', type=str, default='../tum', help='Path to TUM dataset')
        parser.add_argument('--singapore_path', type=str, default='../singapore', help='Path to Singapore dataset')
        parser.add_argument('--aboships_path', type=str, default='../ABOshipsDataset', help='Path to ABOships dataset')
        parser.add_argument('--simuships_path', type=str, default='../simuships', help='Path to Simuships dataset')
        parser.add_argument('--synthia_path', type=str, default='../synthia', help='Path to Synthia dataset')
        parser.add_argument('--nyu_path', type=str, default='../nyu_test/color', help='Path to NYU dataset')
        parser.add_argument('--hypersim_path', type=str, default='../ml-hypersim/contrib/99991/downloads', help='Path to Hypersim dataset')
        
        return parser.parse_args()

    args = parse_args()
    main(args)
