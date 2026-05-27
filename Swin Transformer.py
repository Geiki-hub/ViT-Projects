import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm import create_model
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from PIL import Image
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import multiprocessing

Image.MAX_IMAGE_PIXELS = None
TRAIN_PATH = r"D:\TANG GEI KI\PHD DATASET SPLIT\2. TRAIN DATASET SPLIT\TRAINING (80%)"
VALID_PATH = r"D:\TANG GEI KI\PHD DATASET SPLIT\2. TRAIN DATASET SPLIT\VALIDATION (10%)"
TEST_PATH = r"D:\TANG GEI KI\PHD DATASET SPLIT\2. TRAIN DATASET SPLIT\TESTING (10%)"
OUTPUT_PATH = r'D:\TANG GEI KI\PHD RESULTS\Swin Transformer\Latest (2 Classes)'

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Force GPU
if not torch.cuda.is_available():
    raise RuntimeError("Non-CUDA environment detected. Swin requires a GPU.")
device = torch.device("cuda")

class SwinGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):
        output = self.model(input_tensor)
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        grads = self.gradients.detach()
        acts = self.activations.detach()

        # Swin feature maps are often [B, L, C], needs spatial averaging
        weights = torch.mean(grads, dim=(1), keepdim=True)
        cam = torch.sum(weights * acts, dim=-1).squeeze(0)

        # Reshape to spatial grid (Swin-Base 224 has 7x7 final stage features)
        side = int(np.sqrt(cam.size(0)))
        cam = cam.reshape(side, side)

        cam = F.relu(cam).cpu().numpy()
        return cam

def run_experiment():
    img_size = 224
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(TRAIN_PATH, transform=transform)
    valid_dataset = datasets.ImageFolder(VALID_PATH, transform=transform)
    test_dataset = datasets.ImageFolder(TEST_PATH, transform=transform)
    class_names = train_dataset.classes
    num_classes = len(class_names)  # Should be 2

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    # Load Swin Transformer - SET num_classes=2
    print(f"Initializing Swin for {num_classes} classes...")
    model = create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=num_classes)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    num_epochs = 100
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        t_loss, t_corr, t_total = 0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for imgs, lbls in loop:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            t_loss += loss.item()
            _, pred = outputs.max(1)
            t_total += lbls.size(0)
            t_corr += pred.eq(lbls).sum().item()
            loop.set_postfix(acc=100. * t_corr / t_total)

        # Validation
        model.eval()
        v_loss, v_corr, v_total = 0, 0, 0
        with torch.no_grad():
            for imgs, lbls in valid_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                v_loss += criterion(out, lbls).item()
                v_corr += out.max(1)[1].eq(lbls).sum().item()
                v_total += lbls.size(0)

        history["train_loss"].append(t_loss / len(train_loader))
        history["train_acc"].append(100. * t_corr / t_total)
        history["val_loss"].append(v_loss / len(valid_loader))
        history["val_acc"].append(100. * v_corr / v_total)

        if history["val_loss"][-1] < best_val_loss:
            best_val_loss = history["val_loss"][-1]
            torch.save(model.state_dict(), os.path.join(OUTPUT_PATH, "best_swin.pth"))

    # 1. Loss/Acc Graph
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1);
    plt.plot(history["train_loss"], label="Train");
    plt.plot(history["val_loss"], label="Val");
    plt.legend();
    plt.title("Loss")
    plt.subplot(1, 2, 2);
    plt.plot(history["train_acc"], label="Train");
    plt.plot(history["val_acc"], label="Val");
    plt.legend();
    plt.title("Accuracy")
    plt.savefig(os.path.join(OUTPUT_PATH, "1_accuracy_loss_graph.png"));
    plt.close()

    model.load_state_dict(torch.load(os.path.join(OUTPUT_PATH, "best_swin.pth"), weights_only=True))
    model.eval()

    all_lbls, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            out = model(imgs)
            all_probs.extend(F.softmax(out, dim=1).cpu().numpy())
            all_preds.extend(out.max(1)[1].cpu().numpy())
            all_lbls.extend(lbls.numpy())

    all_lbls, all_preds, all_probs = np.array(all_lbls), np.array(all_preds), np.array(all_probs)

    # 2 & 3. Summary Report
    with open(os.path.join(OUTPUT_PATH, "2_3_results_summary.txt"), "w") as f:
        f.write(classification_report(all_lbls, all_preds, target_names=class_names))

    # 4, 5, 6. Binary ROC/PR Curves
    plt.figure(figsize=(15, 6))

    # ROC (focusing on class 1/positive class)
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(all_lbls, all_probs[:, 1])
    plt.plot(fpr, tpr, color='darkorange', label=f"ROC (AUC={auc(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend();
    plt.title("Binary ROC Curve")

    # PR
    plt.subplot(1, 2, 2)
    p, r, _ = precision_recall_curve(all_lbls, all_probs[:, 1])
    plt.plot(r, p, color='blue', label=f"PR (AP={average_precision_score(all_lbls, all_probs[:, 1]):.2f})")
    plt.legend();
    plt.title("Binary PR Curve")
    plt.savefig(os.path.join(OUTPUT_PATH, "4_5_6_roc_pr_curves.png"));
    plt.close()

    # 7. Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(all_lbls, all_preds), annot=True, fmt='d', xticklabels=class_names,
                yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted');
    plt.ylabel('Actual')
    plt.savefig(os.path.join(OUTPUT_PATH, "7_confusion_matrix.png"));
    plt.close()

    # 8 & 9. Grad-CAM Visualization
    # Target the normalization layer of the last block
    target_layer = model.layers[-1].blocks[-1].norm2
    cam_extractor = SwinGradCAM(model, target_layer)

    sample_img, sample_lbl = test_dataset[0]
    input_tensor = sample_img.unsqueeze(0).to(device)
    heatmap = cam_extractor.generate(input_tensor, sample_lbl)

    heatmap = cv2.resize(heatmap, (img_size, img_size))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    orig_img = sample_img.permute(1, 2, 0).numpy()
    orig_img = (orig_img * 0.229 + 0.485).clip(0, 1)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1);
    plt.imshow(orig_img);
    plt.title("Original Image")
    plt.subplot(1, 2, 2);
    plt.imshow(orig_img);
    plt.imshow(heatmap, cmap='jet', alpha=0.5);
    plt.title("Swin Grad-CAM")
    plt.savefig(os.path.join(OUTPUT_PATH, "8_9_visualizations.png"));
    plt.close()

    print(f"✅ Binary Swin Experiment Complete. Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_experiment()
