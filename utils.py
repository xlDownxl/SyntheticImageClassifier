import torch
from tqdm import tqdm

def compute_fourier(batch_images, log_scale=True, shift=True, normalize=True, apply_highpass= False, highpass_cutoff= 10):
    # Ensure input tensor is in the correct shape (N, 3, H, W)
    assert batch_images.ndim == 4 and batch_images.size(1) == 3, \
        "Input must be a batched tensor of RGB images with shape (N, 3, H, W)"
    
    # Compute 2D Fourier Transform for each channel
    fft_result = torch.fft.fft2(batch_images, dim=(-2, -1))  # Shape: (N, 3, H, W)
    
    # Optionally shift zero-frequency component to the center
    if shift:
        fft_result = torch.fft.fftshift(fft_result, dim=(-2, -1))
    
    if apply_highpass:
        N, C, H, W = fft_result.shape
        
        # Create a meshgrid for radius calculation
        # center = (H/2, W/2) after shift
        yy, xx = torch.meshgrid(
            torch.arange(H, device=batch_images.device),
            torch.arange(W, device=batch_images.device),
            indexing='ij'
        )
        center_y, center_x = (H // 2, W // 2)
        # Euclidean distance from center
        dist_sq = (yy - center_y)**2 + (xx - center_x)**2
        
        # Construct binary mask: 0 inside radius, 1 outside
        # dist_sq < r^2 => zero out
        r_sq = highpass_cutoff**2
        mask = (dist_sq >= r_sq).float()  # shape (H, W)
        
        # Expand to match (N, C, H, W) => just broadcast along N, C
        fft_result = fft_result * mask.unsqueeze(0).unsqueeze(0)
    # Compute the magnitude
    magnitude = torch.abs(fft_result)
    
    # Apply log scaling
    if log_scale:
        magnitude = torch.log1p(magnitude)

    if normalize:
        if apply_highpass:
            channel_means = torch.tensor([6.4653, 6.1062, 5.6956], device=batch_images.device).view(1, 3, 1, 1)
            channel_stds = torch.tensor([4.4637, 4.3371, 3.9269], device=batch_images.device).view(1, 3, 1, 1)
        else:
            channel_means = torch.tensor([-0.9694, -0.9248, -0.9018], device=batch_images.device).view(1, 3, 1, 1)
            channel_stds = torch.tensor([0.2626, 0.2681, 0.2949], device=batch_images.device).view(1, 3, 1, 1)
        
        magnitude = (magnitude - channel_means) / channel_stds
    
    return magnitude

def compute_image_statistics(train_loader, apply_highpass, highpass_cutoff ):
    mean_acc = torch.zeros(3)
    std_acc = torch.zeros(3)
    count = 0

    for batch in tqdm(train_loader):
        # batch shape is (B, 3, H, W) in log-magnitude space
        images = batch[0]  # or batch["image"] etc, depending on how you structure
        # Flatten B, H, W for each channel
        data = compute_fourier(images, log_scale=True, shift=True, apply_highpass = apply_highpass, highpass_cutoff = highpass_cutoff)
        data = data.view(data.size(0), data.size(1), -1)  # shape (B, 3, H*W)
        
        # Compute per-channel mean, std
        batch_mean = data.mean(dim=(0,2))
        batch_std = data.std(dim=(0,2))
        
        mean_acc += batch_mean
        std_acc += batch_std
        count += 1

    global_mean = mean_acc / count
    global_std = std_acc / count

    print("Fourier Mean:", global_mean)
    print("Fourier Std:", global_std)