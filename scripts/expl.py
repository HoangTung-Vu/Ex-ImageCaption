import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
import os

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
        ax.imshow(attn, cmap='Greens', alpha=0.6, extent=(0, image.width, image.height, 0))
        ax.set_title(f"[ {word} ]", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


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

    return image_tensor, caption, attention_maps


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from models.ICtransformer import ICTransformer
    from utils.load import load_config, load_model


    img_path = "data/flickr8k/Images/3545793128_af3af544dc.jpg"
    config_path = "config/train_config1.json"
    config = load_config(config_path)
    checkpoint_path = "checkpoints/best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, vocab = load_model(config, checkpoint_path, device)

    image_tensor, caption, attention_maps = explain_inference_image(img_path, model, vocab)

    # Convert attention maps to numpy arrays for visualization
    attention_maps_np = [attn for attn in attention_maps]

    visualize_attention(image_tensor, caption, attention_maps_np, save_path="attention_visualization.png")