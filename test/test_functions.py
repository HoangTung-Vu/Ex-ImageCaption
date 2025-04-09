import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import math
from models.modules import extract_patches  # Replace with actual module name

def test_extract_patches():
    # Load a random image
    image_path = "data/flickr8k/Images/667626_18933d713e.jpg"  # Replace with actual image path
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension (B=1, C=3, H=32, W=32)
    
    patch_size = 16
    patches = extract_patches(image_tensor, patch_size)

    batch_size, num_patches, patch_dim = patches.shape
    print(f"Extracted patches shape: {patches.shape}")

    grid_size = int(math.sqrt(num_patches))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(6, 6))
    for i in range(num_patches):
        row, col = divmod(i, grid_size)
        patch = patches[0, i].reshape(3, patch_size, patch_size).permute(1, 2, 0)  # Convert to HWC
        axes[row, col].imshow(patch)
        axes[row, col].axis("off")
    plt.show()
    
if __name__ == "__main__":
    test_extract_patches()
