import sys
import os
import torch
from glob import glob
from tqdm import tqdm

# Append upper directory to import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.ICtransformer import ICTransformer
from utils.load import load_config, load_model
from utils.explain import explain_inference_image, visualize_attention

# Paths and config
img_dir = "/home/hoangtungvum/CODE/Explain_Image_Captioning/data/flickr8k/Images"
config_path = "config/train_config3.json"
checkpoint_path = "checkpoints_cetd/best_model.pth"
output_dir = "explain"
os.makedirs(output_dir, exist_ok=True)

# Load model and vocab
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = load_config(config_path)
model, vocab = load_model(config, checkpoint_path, device, model_type="cetd")

# Get 30 image paths
image_paths = sorted(glob(os.path.join(img_dir, "*.jpg")))[:30]

# Process each image
for img_path in tqdm(image_paths, desc="Processing Images"):
    try:
        image_tensor, caption, attention_maps = explain_inference_image(img_path, model, vocab)
        attention_maps_np = [attn for attn in attention_maps]
        
        # Generate a filename-safe output name
        filename = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(output_dir, f"{filename}_attention.png")
        
        # Visualize and save
        visualize_attention(image_tensor, caption, attention_maps_np, save_path=save_path)
    except Exception as e:
        print(f"Failed to process {img_path}: {e}")
