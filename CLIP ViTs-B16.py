import os
import torch
import clip
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import multiprocessing

# -------------------- CONFIG --------------------
TRAIN_PATH = r"D:\TANG GEI KI\PHD DATASET SPLIT\2. TRAIN DATASET SPLIT\TRAINING (80%)"
VAL_PATH = r"D:\TANG GEI KI\PHD DATASET SPLIT\2. TRAIN DATASET SPLIT\VALIDATION (10%)"
TEST_PATH = r"D:\TANG GEI KI\PHD DATASET SPLIT\2. TRAIN DATASET SPLIT\TESTING (10%)"
OUTPUT_DIR = r"D:\TANG GEI KI\PHD RESULTS\CLIP B-16\Latest (2 Classes)"

os.makedirs(OUTPUT_DIR, exist_ok=True)
Image.MAX_IMAGE_PIXELS = None

# Force GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    raise RuntimeError("GPU not found. CLIP-ViT requires CUDA for 100 epochs.")


# -------------------- GRAD-CAM FOR CLIP --------------------
class CLIPGradCAM:
    def __init__(self, model):
        self.model = model.visual
        self.gradients = None
        self.activations = None
        # Target the last transformer block's layer norm
        self.target_layer = self.model.transformer.resblocks[-1].ln_1
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, classifier, class_idx):
        # CLIP Visual Encoder Forward
        # Use .type() to match CLIP's internal precision (float16/float32)
        features = self.model(input_tensor.type(self.model.conv1.weight.dtype))
        output = classifier(features.float())

        classifier.zero_grad()
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        # ViT shape is [Seq, Batch, Dim] -> [197, 1, 768]
        grads = self.gradients.detach()
        acts = self.activations.detach()

        # Pull spatial tokens (excluding CLS token at index 0)
        weights = torch.mean(grads[1:], dim=0)
        cam = torch.sum(weights * acts[1:], dim=-1).squeeze(1)

        # Reshape to 14x14 grid (ViT-B/16: 224/16 = 14)
        cam = F.relu(cam).reshape(14, 14).cpu().numpy()
        return cam


# -------------------- MODEL & DATA --------------------
model_clip, preprocess = clip.load("ViT-B/16", device=device, jit=False)
visual_dim = model_clip.visual.output_dim

# Dataset Loading
train_dataset = datasets.ImageFolder(TRAIN_PATH, transform=preprocess)
val_dataset = datasets.ImageFolder(VAL_PATH, transform=preprocess)
test_dataset = datasets.ImageFolder(TEST_PATH, transform=preprocess)
class_names = train_dataset.classes

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)


class CLIPClassifier(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# Initialize for 2 classes
classifier = CLIPClassifier(visual_dim, len(class_names)).to(device)
optimizer = optim.Adam(classifier.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# -------------------- TRAINING LOOP --------------------
if __name__ == "__main__":
    multiprocessing.freeze_support()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    EPOCHS = 100
    best_acc = 0

    for epoch in range(1, EPOCHS + 1):
        classifier.train()
        t_loss, t_corr, t_total = 0, 0, 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for imgs, lbls in loop:
            imgs, lbls = imgs.to(device), lbls.to(device)

            with torch.no_grad():
                # CLIP image encoder is usually kept frozen
                features = model_clip.encode_image(imgs).float()

            optimizer.zero_grad()
            outputs = classifier(features)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            t_loss += loss.item()
            t_corr += outputs.argmax(1).eq(lbls).sum().item()
            t_total += lbls.size(0)
            loop.set_postfix(acc=100. * t_corr / t_total)

        # Validation
        classifier.eval()
        v_loss, v_corr, v_total = 0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                feats = model_clip.encode_image(imgs).float()
                out = classifier(feats)
                v_loss += criterion(out, lbls).item()
                v_corr += out.argmax(1).eq(lbls).sum().item()
                v_total += lbls.size(0)

        history["train_loss"].append(t_loss / len(train_loader))
        history["train_acc"].append(100. * t_corr / t_total)
        history["val_loss"].append(v_loss / len(val_loader))
        history["val_acc"].append(100. * v_corr / v_total)

        if history["val_acc"][-1] > best_acc:
            best_acc = history["val_acc"][-1]
            torch.save(classifier.state_dict(), os.path.join(OUTPUT_DIR, 'best_clip_classifier.pth'))

    # -------------------- EVALUATION & PLOTTING --------------------
    # 1. Graph
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1);
    plt.plot(history["train_loss"], label='Train');
    plt.plot(history["val_loss"], label='Val');
    plt.title("Loss");
    plt.legend()
    plt.subplot(1, 2, 2);
    plt.plot(history["train_acc"], label='Train');
    plt.plot(history["val_acc"], label='Val');
    plt.title("Accuracy");
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_accuracy_loss_graph.png"));
    plt.close()

    classifier.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_clip_classifier.pth'), weights_only=True))
    classifier.eval()

    all_lbls, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            feats = model_clip.encode_image(imgs).float()
            out = classifier(feats)
            all_probs.extend(F.softmax(out, dim=1).cpu().numpy())
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_lbls.extend(lbls.numpy())

    all_lbls, all_preds, all_probs = np.array(all_lbls), np.array(all_preds), np.array(all_probs)

    # 2 & 3. Summary
    with open(os.path.join(OUTPUT_DIR, "2_3_classification_report.txt"), "w") as f:
        f.write(classification_report(all_lbls, all_preds, target_names=class_names))

    # 4, 5, 6. Binary Curves
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(all_lbls, all_probs[:, 1])  # Positive class probability
    plt.plot(fpr, tpr, color='darkorange', label=f"ROC (AUC={auc(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], 'k--');
    plt.title("ROC Curve");
    plt.legend()

    plt.subplot(1, 2, 2)
    p, r, _ = precision_recall_curve(all_lbls, all_probs[:, 1])
    plt.plot(r, p, color='blue', label=f"PR (AP={average_precision_score(all_lbls, all_probs[:, 1]):.2f})")
    plt.title("PR Curve");
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "4_5_6_roc_pr_metrics.png"));
    plt.close()

    # 7. Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(all_lbls, all_preds), annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.savefig(os.path.join(OUTPUT_DIR, "7_confusion_matrix.png"));
    plt.close()

    # 8 & 9. Grad-CAM Visualization
    cam_gen = CLIPGradCAM(model_clip)
    sample_img, sample_lbl = test_dataset[0]
    input_tensor = sample_img.unsqueeze(0).to(device)

    # Generate CAM map
    cam_map = cam_gen.generate(input_tensor, classifier, sample_lbl)
    cam_map = cam_map.astype(np.float32)  # Fix for cv2.resize
    cam_map = cv2.resize(cam_map, (224, 224))

    # Reverse normalization for visualization
    inv_normalize = transforms.Normalize(
        mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711],
        std=[1 / 0.26862954, 1 / 0.26130258, 1 / 0.27577711]
    )
    plot_img = inv_normalize(sample_img).permute(1, 2, 0).cpu().numpy().clip(0, 1)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1);
    plt.imshow(plot_img);
    plt.title("Original")
    plt.subplot(1, 2, 2);
    plt.imshow(plot_img);
    plt.imshow(cam_map, cmap='jet', alpha=0.5);
    plt.title("CLIP Grad-CAM")
    plt.savefig(os.path.join(OUTPUT_DIR, "8_9_visualizations.png"));
    plt.close()

    print(f"✅ CLIP Binary Classification Complete. Results in: {OUTPUT_DIR}")
