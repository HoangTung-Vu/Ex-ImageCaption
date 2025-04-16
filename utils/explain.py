import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import math

def visualize_attention(image_tensor, caption, attention_maps, save_path=None):
    """
    Visualizes attention maps over the image for each word in the caption.

    Args:
        image_tensor (torch.Tensor): Input image tensor [1, 3, H, W]
        caption (List[str]): List of generated words
        attention_maps (List[np.ndarray]): Corresponding attention maps (H', W')
        save_path (str): Optional, save the output figure
    """
    # Convert image tensor to displayable image
    image = TF.to_pil_image(image_tensor.squeeze(0).cpu())

    num_words = len(caption)
    fig, axs = plt.subplots(1, num_words + 1, figsize=(2 * (num_words + 1), 4))

    axs[0].imshow(image)
    axs[0].axis("off")
    axs[0].set_title("Input")

    for i, (word, attn) in enumerate(zip(caption, attention_maps)):
        ax = axs[i + 1]
        ax.imshow(image)
        ax.imshow(attn, alpha=0.5, extent=(0, image.width, image.height, 0))
        ax.set_title(f"[ {word} ]", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def reconstruct_heatmap(conv_layers, rf_map):
    """
    Dùng cho mô hình sử dụng Conv
    Args:
        conv_layers: list of nn.Conv2d (stride >= 1)
        rf_map: Tensor, shape (B,1,H',W'), receptive field heatmap (1 channel)
    Returns:
        heatmap: Tensor, shape (B,1,H, W), phóng ngược về kích thước gốc nhờ upsample + smoothing
    """
    x = rf_map
    for conv in reversed(conv_layers):
        stride = conv.stride[0] if isinstance(conv.stride, (tuple, list)) else conv.stride
        x = F.interpolate(x, scale_factor=stride, mode='bilinear', align_corners=False)

        k = conv.kernel_size[0] if isinstance(conv.kernel_size, (tuple, list)) else conv.kernel_size
        smoothing_kernel = torch.ones(1, 1, k, k, device=x.device) / (k * k)
        x = F.conv2d(x, weight=smoothing_kernel, padding=k // 2)
        
    return x


def explain_inference_image(img_path: str, model, vocab, device='cuda'):
    """
    Load an image, preprocess it, and run the model to generate caption + attention maps.

    Args:
        img_path (str): Path to the image
        model: Trained ICTransformer model
        vocab: Vocabulary object with 'stoi' and 'itos'
        device (str): 'cuda' or 'cpu'

    Returns:
        Tuple[List[str], List[np.ndarray]]: caption words and attention maps
    """

    # Define image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load and preprocess image
    image = Image.open(img_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # shape: [1, 3, 224, 224]

    # Generate caption and attention maps
    caption, attention_maps = model.caption_image_greedy(image_tensor, vocab)
    modelname = model.__class__.__name__
    if modelname == 'ICTransformer2' or modelname == 'ICTransformer' or modelname == 'ConvEnTransDe':
        grid_size = int(math.sqrt(attention_maps[0].size(0)))

        heatmap = [attention_vector.view(grid_size, grid_size).detach().cpu().numpy() for attention_vector in attention_maps]
    elif modelname == 'CvT_IC' : 
        grid_size = int(math.sqrt(attention_maps[0].size(0)))
        conv_layers = [
            nn.Conv2d(kernel_size=7, in_channels=3, out_channels=64, stride=4, padding=3), 
            nn.Conv2d(kernel_size=3, in_channels=64, out_channels=192, stride=2, padding=1), 
            nn.Conv2d(kernel_size=3, in_channels=129, out_channels=384, stride=2, padding=1),
        ]
        heatmap = [
            reconstruct_heatmap(
                conv_layers,
                attention_vector.view(1, 1, grid_size, grid_size)  # Shape: [B=1, C=1, H, W]
            ).squeeze(0).squeeze(0).detach().cpu().numpy()  # Remove batch and channel dim for visualization
            for attention_vector in attention_maps
        ]


    else:
        return "Model not supported for explanation"
    return image_tensor, caption, heatmap


