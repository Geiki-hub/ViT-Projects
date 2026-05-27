import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import seaborn as sns
import numpy as np
import os
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_channels=3, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.att_weights = None

    def forward(self, x):
        attn_output, weights = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        self.att_weights = weights
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=128, patch_size=16, num_classes=2, embed_dim=128, num_heads=4, depth=4, mlp_dim=256):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_encoding = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp_head = nn.Linear(embed_dim, num_classes)
        self.gradients = None

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_encoding
        for block in self.transformer_blocks:
            x = block(x)

        if x.requires_grad:
            x.register_hook(lambda grad: setattr(self, 'gradients', grad))

        self.last_activations = x
        return self.mlp_head(self.norm(x[:, 0]))


def train_model(train_path, valid_path, test_path, output_path, epochs=100, batch_size=32):
    os.makedirs(output_path, exist_ok=True)

    if not torch.cuda.is_available():
        print("Warning: GPU not found. Running on CPU (this will be very slow).")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Loading Datasets
    train_data = datasets.ImageFolder(train_path, transform=transform)
    valid_data = datasets.ImageFolder(valid_path, transform=transform)
    test_data = datasets.ImageFolder(test_path, transform=transform)
    class_names = train_data.classes
    num_classes = len(class_names)
    print(f"Detected Classes: {class_names}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

    model = VisionTransformer(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

        # Validation
        model.eval()
        val_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, pred = outputs.max(1)
                v_total += labels.size(0)
                v_correct += pred.eq(labels).sum().item()

        history["train_loss"].append(train_loss / len(train_loader))
        history["train_acc"].append(100. * correct / total)
        history["val_loss"].append(val_loss / len(valid_loader))
        history["val_acc"].append(100. * v_correct / v_total)

        if history["val_loss"][-1] < best_val_loss:
            best_val_loss = history["val_loss"][-1]
            torch.save(model.state_dict(), os.path.join(output_path, "best_vit.pth"))

    # Plot Accuracy/Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label='Train')
    plt.plot(history["val_loss"], label='Val')
    plt.title('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label='Train')
    plt.plot(history["val_acc"], label='Val')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_path, "1_accuracy_loss_graph.png"))
    plt.close()

    # Final Evaluation on Test Set
    model.load_state_dict(torch.load(os.path.join(output_path, "best_vit.pth"), weights_only=True))
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, pred = outputs.max(1)
            all_labels.extend(labels.numpy())
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    # Save Classification Report
    with open(os.path.join(output_path, "2_3_final_results_summary.txt"), "w") as f:
        f.write(classification_report(all_labels, all_preds, target_names=class_names))

    # ROC and PR Curves (Optimized for Binary)
    plt.figure(figsize=(15, 6))

    # ROC Curve
    plt.subplot(1, 2, 1)
    # For binary classification, we focus on the probability of the positive class (index 1)
    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title("Receiver Operating Characteristic (Binary)")
    plt.legend(loc="lower right")

    # PR Curve
    plt.subplot(1, 2, 2)
    precision, recall, _ = precision_recall_curve(all_labels, all_probs[:, 1])
    ap = average_precision_score(all_labels, all_probs[:, 1])
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR (AP = {ap:.2f})')
    plt.title("Precision-Recall Curve (Binary)")
    plt.legend(loc="lower left")

    plt.savefig(os.path.join(output_path, "4_5_6_roc_pr_metrics.png"))
    plt.close()

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(all_labels, all_preds), annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(output_path, "7_confusion_matrix.png"))
    plt.close()

    # Visualizations (GradCAM & Rollout)
    sample_img, sample_lbl = test_data[0]
    input_tensor = sample_img.unsqueeze(0).to(device).requires_grad_(True)

    model.zero_grad()
    out = model(input_tensor)
    out[0, sample_lbl].backward()

    grads = model.gradients
    acts = model.last_activations.detach()
    weights = torch.mean(grads, dim=1, keepdim=True)
    cam = F.relu(torch.sum(weights * acts, dim=-1))[:, 1:].reshape(8, 8).cpu().numpy()
    cam = cv2.resize(cam, (128, 128))

    # Attention Rollout
    num_tokens = model.transformer_blocks[0].att_weights.shape[-1]
    eye = torch.eye(num_tokens).to(device)
    res = eye
    for block in model.transformer_blocks:
        weights = block.att_weights.mean(1).squeeze(0)
        res = torch.matmul(0.5 * weights + 0.5 * eye, res)

    rollout = res[0, 1:].reshape(8, 8).detach().cpu().numpy()
    rollout = cv2.resize(rollout, (128, 128))

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(sample_img.permute(1, 2, 0) * 0.5 + 0.5)
    plt.title("Original Image")
    plt.subplot(1, 3, 2)
    plt.imshow(cam, cmap='jet')
    plt.title("GradCAM")
    plt.subplot(1, 3, 3)
    plt.imshow(rollout, cmap='inferno')
    plt.title("Attention Rollout")
    plt.savefig(os.path.join(output_path, "8_9_visualizations.png"))
    plt.close()


if __name__ == "__main__":
    train_model(
        train_path=r"D:\TANG GEI KI\PHD DATASET SPLIT\2. TRAIN DATASET SPLIT\TRAINING (80%)",
        valid_path=r"D:\TANG GEI KI\PHD DATASET SPLIT\2. TRAIN DATASET SPLIT\VALIDATION (10%)",
        test_path=r"D:\TANG GEI KI\PHD DATASET SPLIT\2. TRAIN DATASET SPLIT\TESTING (10%)",
        output_path=r"D:\TANG GEI KI\PHD RESULTS\ViTs Pretrained\Latest (2 classes)",
        epochs=100
    )
