import torch
from torchvision import transforms
from PIL import Image
from torchmetrics.functional import structural_similarity_index_measure as ssim

# Define transform: resize + grayscale + to tensor
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),  # Converts to float tensor in [0,1]
])

# Load images from paths
img_pred = Image.open("40_real_B.png").convert("L")  # L = grayscale
img_target = Image.open("40_real_B.png").convert("L")

# Apply transforms
pred_tensor = transform(img_pred).unsqueeze(0)  # shape: (1, 1, H, W)
target_tensor = transform(img_target).unsqueeze(0)

# Compute SSIM
score = ssim(pred_tensor, target_tensor)
print(f"SSIM score: {score.item():.4f}")

import torch
from torchvision import transforms
from PIL import Image

# Function to compute Dice Score
def dice_score(pred, target, threshold=0.5):
    # Binarize the predicted image
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()
    
    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum()
    dice = (2. * intersection) / (union + 1e-8)
    return dice.item()

# Image loading and preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),  # Converts to [0,1] float tensor
])

# Load predicted and ground truth images (grayscale)
pred_img = Image.open("40_real_B.png").convert("L")
target_img = Image.open("40_real_B.png").convert("L")

# Transform to tensors
pred_tensor = transform(pred_img).squeeze(0)   # shape: (H, W)
target_tensor = transform(target_img).squeeze(0)

# Compute Dice Score
score = dice_score(pred_tensor, target_tensor, threshold=0.5)
print(f"Dice Score: {score:.4f}")
