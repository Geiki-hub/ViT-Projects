import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from tqdm import tqdm
import cv2
from PIL import Image


# ----------------------------- Model Components -----------------------------

class SoftSplit(nn.Module):
    def __init__(self, in_ch=3, patch_size=7, stride=4, proj_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, proj_dim, kernel_size=patch_size, stride=stride, padding=patch_size // 2)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2)


class TokenTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        res = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = res + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class T2TModule(nn.Module):
    def __init__(self, in_ch=3, token_dim=64):
        super().__init__()
        # First soft split: 224 -> (224/4) = 56
        self.soft_split1 = SoftSplit(in_ch, patch_size=7, stride=4, proj_dim=32)
        self.trans1 = TokenTransformerBlock(dim=32)
        # Second soft split: 56 -> (56/2) = 28
        self.soft_split2 = SoftSplit(32, patch_size=3, stride=2, proj_dim=token_dim)

    def forward(self, x):
        x = self.soft_split1(x)
        x = self.trans1(x)
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.soft_split2(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, num_classes=2, token_dim=64, depth=4):
        super().__init__()
        self.t2t = T2TModule(in_ch=3, token_dim=token_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        # Initial pos_embed size; will be updated in forward if needed
        self.pos_embed = nn.Parameter(torch.zeros(1, 1000, token_dim))
        self.blocks = nn.ModuleList([TokenTransformerBlock(dim=token_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(token_dim)
        self.head = nn.Linear(token_dim, num_classes)
        self.gradients = None

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.t2t(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if x.size(1) > self.pos_embed.size(1):
            new_pos = torch.zeros(1, x.size(1), x.size(2), device=x.device)
            nn.init.trunc_normal_(new_pos, std=0.02)
            self.pos_embed = nn.Parameter(new_pos)

        x = x + self.pos_embed[:, :x.size(1), :]
        for block in self.blocks:
            x = block(x)

        if x.requires_grad:
            x.register_hook(lambda grad: setattr(self, 'gradients', grad))

        self.last_activations = x
        x = self.norm(x)
        return self.head(x[:, 0])


# ----------------------------- Main Execution -----------------------------

def run_t2t_experiment():
    TRAIN_PATH = r'D:\TANG GEI KI\PHD DATASET SPLIT\2. TRAIN DATASET SPLIT\TRAINING (80%)'
    VALID_PATH = r'D:\TANG GEI KI\PHD DATASET SPLIT\2. TRAIN DATASET SPLIT\VALIDATION (10%)'
    TEST_PATH = r'D:\TANG GEI KI\PHD DATASET SPLIT\2. TRAIN DATASET SPLIT\TESTING (10%)'
    OUTPUT_DIR = r'D:\TANG GEI KI\PHD Results\T2T ViTs\Latest (2 Classes)'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 100
    BATCH_SIZE = 8

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(TRAIN_PATH, transform=transform)
    valid_data = datasets.ImageFolder(VALID_PATH, transform=transform)
    test_data = datasets.ImageFolder(TEST_PATH, transform=transform)
    class_names = train_data.classes
    num_classes = len(class_names)  # Should be 2

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    model = VisionTransformer(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # --- Training Loop ---
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss, t_acc = 0, 0
        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch}"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            t_acc += out.argmax(1).eq(lbls).sum().item()

        # Validation
        model.eval()
        v_loss, v_acc = 0, 0
        with torch.no_grad():
            for imgs, lbls in valid_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                v_loss += criterion(out, lbls).item()
                v_acc += out.argmax(1).eq(lbls).sum().item()

        history["train_loss"].append(t_loss / len(train_loader))
        history["train_acc"].append(100. * t_acc / len(train_data))
        history["val_loss"].append(v_loss / len(valid_loader))
        history["val_acc"].append(100. * v_acc / len(valid_data))

    # --- Plots & Metrics ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1);
    plt.plot(history['train_loss'], label='Train');
    plt.plot(history['val_loss'], label='Val');
    plt.title("Loss");
    plt.legend()
    plt.subplot(1, 2, 2);
    plt.plot(history['train_acc'], label='Train');
    plt.plot(history['val_acc'], label='Val');
    plt.title("Accuracy");
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_accuracy_loss.png"));
    plt.close()

    model.eval()
    all_lbls, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device);
            out = model(imgs)
            all_probs.extend(F.softmax(out, dim=1).cpu().numpy())
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_lbls.extend(lbls.numpy())

    all_lbls, all_preds, all_probs = np.array(all_lbls), np.array(all_preds), np.array(all_probs)

    # Classification Report
    with open(os.path.join(OUTPUT_DIR, "2_3_report.txt"), "w") as f:
        f.write(classification_report(all_lbls, all_preds, target_names=class_names))

    # Binary Curves (Optimized for 2 classes)
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(all_lbls, all_probs[:, 1])
    plt.plot(fpr, tpr, color='darkorange', label=f"ROC (AUC={auc(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], 'k--');
    plt.title("Binary ROC Curve");
    plt.legend()

    plt.subplot(1, 2, 2)
    p, r, _ = precision_recall_curve(all_lbls, all_probs[:, 1])
    plt.plot(r, p, color='blue', label=f"PR (AP={average_precision_score(all_lbls, all_probs[:, 1]):.2f})")
    plt.title("Binary PR Curve");
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "4_5_curves.png"));
    plt.close()

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(all_lbls, all_preds), annot=True, fmt='d', xticklabels=class_names,
                yticklabels=class_names, cmap='Blues')
    plt.savefig(os.path.join(OUTPUT_DIR, "7_confusion_matrix.png"));
    plt.close()

    # Grad-CAM
    sample_img, sample_lbl = test_data[0]
    input_tensor = sample_img.unsqueeze(0).to(device).requires_grad_(True)
    model.zero_grad();
    output = model(input_tensor);
    output[0, sample_lbl].backward()

    weights = torch.mean(model.gradients, dim=1, keepdim=True)
    # T2T resolution for 224 input is usually 28x28 tokens + 1 CLS token
    num_spatial_tokens = model.last_activations.size(1) - 1
    grid_size = int(np.sqrt(num_spatial_tokens))

    cam = F.relu(torch.sum(weights * model.last_activations.detach(), dim=-1))[:, 1:].reshape(grid_size,
                                                                                              grid_size).cpu().numpy()
    cam = cv2.resize(cam, (224, 224))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    # Display unnormalized image
    plt.imshow(sample_img.permute(1, 2, 0).numpy() * 0.229 + 0.485)
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(cam, cmap='jet')
    plt.title("T2T Grad-CAM")
    plt.savefig(os.path.join(OUTPUT_DIR, "8_9_gradcam.png"));
    plt.close()

    print(f"✅ T2T Binary success! Results in: {OUTPUT_DIR}")


if __name__ == '__main__':
    run_t2t_experiment()