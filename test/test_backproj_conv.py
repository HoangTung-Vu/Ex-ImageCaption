import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def reconstruct_heatmap(conv_layers, rf_map):
    """
    Args:
        conv_layers: list of nn.Conv2d (stride >= 1)
        rf_map: Tensor, shape (B,1,H',W'), receptive field heatmap (1 channel)
    Returns:
        heatmap: Tensor, shape (B,1,H, W), phóng ngược về kích thước gốc nhờ upsample + smoothing
    """
    x = rf_map
    # Up-sample lần lượt qua từng tầng conv (theo chiều ngược)
    for conv in reversed(conv_layers):
        # Lấy stride (int hoặc tuple)
        stride = conv.stride[0] if isinstance(conv.stride, (tuple, list)) else conv.stride
        # 1) Nội suy bilinear để upsample
        x = F.interpolate(x, scale_factor=stride, mode='bilinear', align_corners=False)
        # 2) Làm mịn với kernel trung bình 3×3 (có thể điều chỉnh kích thước nếu cần)
        smoothing_kernel = torch.ones(1, 1, 3, 3, device=x.device) / 9.0
        x = F.conv2d(x, weight=smoothing_kernel, padding=1)
    return x


if __name__ == '__main__':
    # Path to an example image
    img_path = '/home/hoangtungvum/CODE/Explain_Image_Captioning/data/flickr8k/Images/10815824_2997e03d76.jpg'
    img = Image.open(img_path).convert('RGB')

    # Resize ảnh để đơn giản hóa conv
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0)  # Shape: (1, 3, 128, 128)

    # Định nghĩa các conv_layers giả định
    conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
    conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
    conv_layers = [conv1, conv2]

    # Tạo rf_map giả lập (1 channel, 32x32) với random noise hoặc highlight
    # rf_map = torch.randn(1, 1, 32, 32)
    rf_map = torch.zeros(1, 1, 32, 32); 
    rf_map[0, 0, 15:17, 15:17] = 1.0
    rf_map[0,0, 25:27, 25:27] = 0.5
    rf_map[0,0, 10:20, 20:25] = 0.5


    # Reconstruct heatmap
    heatmap = reconstruct_heatmap(conv_layers, rf_map)  # Output: (1, 1, 128, 128)

    # Normalize để hiển thị
    def normalize_tensor(t):
        t = t.squeeze()  # (H, W)
        t = t - t.min()
        t = t / (t.max() + 1e-6)
        return t.detach().cpu().numpy()

    img_np = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    rf_np = normalize_tensor(rf_map)
    heatmap_np = normalize_tensor(heatmap)

    # Hiển thị
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(rf_np, cmap='hot')
    plt.title('RF Map (downsampled)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img_np)
    plt.imshow(heatmap_np, cmap='jet', alpha=0.5)
    plt.title('Reconstructed Heatmap Overlay')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
