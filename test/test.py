import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

# Load ảnh từ URL
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert('RGB')

# Tiền xử lý ảnh
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
])
input_tensor = transform(image).unsqueeze(0)  # shape: (1, 3, 224, 224)

# Load ResNet-50 pretrained
model = models.resnet50(pretrained=True)
model.eval()

# Hook để lấy feature map từ layer cuối
features = []
def hook(module, input, output):
    features.append(output)

model.layer4.register_forward_hook(hook)

# Forward để lấy prediction
with torch.no_grad():
    output = model(input_tensor)  # logits

# Lấy class dự đoán top-1
pred_class = output.argmax(dim=1).item()

# === CAM TRUYỀN THỐNG === #
feature_map = features[0].squeeze(0)  # shape: (2048, 7, 7)
fc_weights = model.fc.weight  # shape: (1000, 2048)
class_weights = fc_weights[pred_class]  # shape: (2048,)

# Tính CAM theo không gian 7x7
cam = torch.sum(feature_map * class_weights.view(-1, 1, 1), dim=0)  # shape: (7, 7)
cam = cam.detach().cpu().numpy()
cam = np.maximum(cam, 0)
cam = cam - cam.min()
cam = cam / (cam.max() + 1e-8)
cam_resized = Image.fromarray(np.uint8(cam * 255)).resize((224, 224), resample=Image.BILINEAR)
cam_resized = np.array(cam_resized)

# === CAM THEO CHIỀU CHANNEL === #
# Global Average Pooling để lấy đặc trưng (2048,)
gap_features = F.adaptive_avg_pool2d(feature_map.unsqueeze(0), 1).squeeze()  # (2048,)
channel_contributions = gap_features * class_weights  # (2048,)
channel_contrib_np = channel_contributions.detach().cpu().numpy()
channel_contrib_norm = (channel_contrib_np - channel_contrib_np.min()) / (channel_contrib_np.max() - channel_contrib_np.min() + 1e-8)

# === Visualization === #
plt.figure(figsize=(14, 4))

# Ảnh gốc
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

# CAM spatial
plt.subplot(1, 3, 2)
plt.imshow(image)
plt.imshow(cam_resized, cmap='jet', alpha=0.5)
plt.title(f"CAM Spatial (Class {pred_class})")
plt.axis('off')

# Biểu đồ channel-wise contribution
plt.subplot(1, 3, 3)
plt.plot(channel_contrib_norm)
plt.title(f"Channel Contributions (Class {pred_class})")
plt.xlabel("Channel index (0–2047)")
plt.ylabel("Normalized Contribution")
plt.grid(True)

plt.tight_layout()
plt.show()

# === Hiển thị 5 feature map có ảnh hưởng lớn nhất === #
topk = channel_contrib_np.argsort()[-5:][::-1]  # Top 5 kênh có ảnh hưởng lớn nhất
for i in topk:
    fmap = feature_map[i].cpu().numpy()
    fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
    plt.imshow(fmap, cmap='viridis')
    plt.title(f"Top Channel {i} - Contribution Score: {channel_contrib_np[i]:.4f}")
    plt.axis('off')
    plt.show()

# === Hiển thị 10 feature maps đầu tiên sau khi upsample về 224x224 === #
upsampled_maps = F.interpolate(feature_map.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
upsampled_maps = upsampled_maps.squeeze(0).cpu().numpy()  # shape: (2048, 224, 224)
# === Overlay 10 feature maps đầu tiên lên ảnh gốc như heatmap === #
plt.figure(figsize=(15, 6))
for idx in range(10):
    fmap = upsampled_maps[idx]
    fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
    
    # Resize lại ảnh gốc nếu cần (nên đã là 224x224 rồi)
    image_np = np.array(image.resize((224, 224)))

    # Tạo heatmap overlay
    heatmap = plt.cm.jet(fmap)[:, :, :3]  # RGB, bỏ alpha
    overlay = (0.5 * image_np / 255.0 + 0.5 * heatmap)
    overlay = np.clip(overlay, 0, 1)

    plt.subplot(2, 5, idx + 1)
    plt.imshow(overlay)
    plt.title(f"Overlay Channel {idx}")
    plt.axis('off')

plt.suptitle("Overlay 10 Feature Maps on Original Image", fontsize=16)
plt.tight_layout()
plt.show()
