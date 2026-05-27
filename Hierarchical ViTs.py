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


# ----------------------------- Hierarchical Model Components -----------------------------

class HierarchicalBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        res = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = res + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.reduction = nn.Linear(in_dim * 4, out_dim)
        self.norm = nn.LayerNorm(in_dim * 4)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, (H + 1) // 2, (W + 1) // 2


class HierarchicalViT(nn.Module):
    def __init__(self, num_classes=2, embed_dim=64):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=4, stride=4)
        self.stage1 = nn.ModuleList([HierarchicalBlock(embed_dim, 4) for _ in range(2)])
        self.merge1 = PatchMerging(embed_dim, embed_dim * 2)
        self.stage2 = nn.ModuleList([HierarchicalBlock(embed_dim * 2, 8) for _ in range(2)])
        self.merge2 = PatchMerging(embed_dim * 2, embed_dim * 4)
        self.stage3 = nn.ModuleList([HierarchicalBlock(embed_dim * 4, 16) for _ in range(2)])
        self.norm = nn.LayerNorm(embed_dim * 4)
        self.head = nn.Linear(embed_dim * 4, num_classes)
        self.gradients = None

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        for blk in self.stage1: x = blk(x)
        x, H, W = self.merge1(x, H, W)
        for blk in self.stage2: x = blk(x)
        x, H, W = self.merge2(x, H, W)
        for blk in self.stage3: x = blk(x)
        if x.requires_grad:
            x.register_hook(lambda grad: setattr(self, 'gradients', grad))
        self.last_activations = x
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


# ----------------------------- Execution -----------------------------

def run_hvit_experiment():
    TRAIN_PATH = r'D:\TANG GEI KI\PHD DATASET SPLIT\2. TRAIN DATASET SPLIT\TRAINING (80%)'
    VALID_PATH = r'D:\TANG GEI KI\PHD DATASET SPLIT\2. TRAIN DATASET SPLIT\VALIDATION (10%)'
    TEST_PATH = r'D:\TANG GEI KI\PHD DATASET SPLIT\2. TRAIN DATASET SPLIT\TESTING (10%)'
    OUTPUT_DIR = r'D:\TANG GEI KI\PHD Results\Hierarchical ViT\Full_Metrics'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Force GPU Only
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not found.")
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True

    EPOCHS = 100
    BATCH_SIZE = 16
    history = {'train_loss': [], 'train_acc': []}

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(TRAIN_PATH, transform=transform)
    test_data = datasets.ImageFolder(TEST_PATH, transform=transform)
    class_names = train_data.classes

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    model = HierarchicalViT(num_classes=len(class_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # --- Training ---
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss, t_acc, t_total = 0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}")
        for imgs, lbls in loop:
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()

            t_loss += loss.item()
            t_acc += out.argmax(1).eq(lbls).sum().item()
            t_total += lbls.size(0)
            loop.set_postfix(acc=100. * t_acc / t_total)

        history['train_loss'].append(t_loss / len(train_loader))
        history['train_acc'].append(100. * t_acc / t_total)

    # --- 1. Accuracy & Loss Graphs ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Loss', color='red')
    plt.title('Training Loss');
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Accuracy', color='green')
    plt.title('Training Accuracy');
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_accuracy.png"));
    plt.close()

    # --- Evaluation ---
    model.eval()
    all_lbls, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            out = model(imgs)
            all_probs.extend(F.softmax(out, dim=1).cpu().numpy())
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_lbls.extend(lbls.numpy())

    all_lbls, all_preds, all_probs = np.array(all_lbls), np.array(all_preds), np.array(all_probs)

    # --- 2. Confusion Matrix ---
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(all_lbls, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual');
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"));
    plt.close()

    # Classification Report
    with open(os.path.join(OUTPUT_DIR, "report.txt"), "w") as f:
        f.write(classification_report(all_lbls, all_preds, target_names=class_names))

    # Binary Curves
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(all_lbls, all_probs[:, 1])
    plt.plot(fpr, tpr, color='darkorange', label=f"AUC={auc(fpr, tpr):.2f}")
    plt.plot([0, 1], [0, 1], 'k--');
    plt.legend();
    plt.title("ROC Curve")

    plt.subplot(1, 2, 2)
    p, r, _ = precision_recall_curve(all_lbls, all_probs[:, 1])
    plt.plot(r, p, color='blue', label=f"AP={average_precision_score(all_lbls, all_probs[:, 1]):.2f}")
    plt.legend();
    plt.title("PR Curve")
    plt.savefig(os.path.join(OUTPUT_DIR, "metrics.png"));
    plt.close()

    # Grad-CAM logic remains same...
    sample_img, sample_lbl = test_data[0]
    input_tensor = sample_img.unsqueeze(0).to(device).requires_grad_(True)
    model.zero_grad();
    out = model(input_tensor);
    out[0, sample_lbl].backward()
    weights = torch.mean(model.gradients, dim=1, keepdim=True)
    cam = F.relu(torch.sum(weights * model.last_activations.detach(), dim=-1)).reshape(14, 14).cpu().numpy()
    cam = cv2.resize(cam, (224, 224))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1);
    plt.imshow(sample_img.permute(1, 2, 0).numpy() * 0.2 + 0.5);
    plt.title("Original")
    plt.subplot(1, 2, 2);
    plt.imshow(cam, cmap='jet');
    plt.title("Grad-CAM")
    plt.savefig(os.path.join(OUTPUT_DIR, "gradcam.png"));
    plt.close()

    print(f"✅ Success! Graphs and CM saved in: {OUTPUT_DIR}")


if __name__ == '__main__':
    run_hvit_experiment()
