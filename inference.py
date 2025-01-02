import torch
import torch.nn as nn
from torchvision import models
from transforms import *
import argparse
from utils import *

def define_model(num_classes=2, pretrained=True):
    model = models.resnet18(pretrained=pretrained)

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.Linear(256, num_classes)
    )  

    return model

def main(args):
    model_path = args.model_path

    test_transform = get_test_transforms(
        img_size=args.img_size, 
        use_gradient=args.use_gradient, 
        use_fourier=args.use_fourier, 
        crop=args.crop
    )

    model = define_model(num_classes=2, pretrained=False) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.load_state_dict(torch.load(model_path))
    print(f"Loaded model from {model_path} for inference")

    model.to(device)
    model.eval()

    image = Image.open(args.image_path).convert('RGB')

    images = test_transform(image)
    images = images.to(device).unsqueeze(0)

    fourier_image = compute_fourier(images, log_scale=True, shift=True, apply_highpass=args.apply_highpass, highpass_cutoff=args.highpass_cutoff)

    output = model(fourier_image)
    print(f"Prediction for image at {args.image_path}:")
    #print(output)
    _, preds = torch.max(output, 1)
    pred_class = 'Synthetic' if preds.item() == 1 else 'Real'
    print("***"+pred_class+"***")


if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser(description="Train and evaluate a maritime classifier.")
        parser.add_argument('--model_path', type=str, default='resnet18_synthetic_classifier', help='Path to the trained model')
        parser.add_argument('--image_path', type=str, default='resnet18_synthetic_classifier', help='Path to the image')

        parser.add_argument('--crop', action='store_true', help='Crop instead of resize')
        parser.add_argument('--img_size', type=int, nargs=2, default=[640,480], help="Image size for training and evaluation")

        parser.add_argument('--use_gradient', action='store_true', help='Use gradient transform')
        parser.add_argument('--use_fourier', action='store_true', help='Use Fourier transform')
        parser.add_argument('--apply_highpass', action='store_true', help='Apply high-pass filter')
        parser.add_argument('--highpass_cutoff', type=int, default=10, help='High-pass filter cutoff frequency')

        return parser.parse_args()

    args = parse_args()
    main(args)
