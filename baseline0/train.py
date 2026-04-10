import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)

from dataset import get_dataloaders
from ITCS352_DL_Project.baseline0.model import SimpleCNN


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("outputs_belgiumts")
OUTPUT_DIR.mkdir(exist_ok=True)

NUM_EPOCHS = 15
LEARNING_RATE = 0.001
BATCH_SIZE = 32
IMG_SIZE = 64
EARLY_STOPPING_PATIENCE = 5


def evaluate_model(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, acc, macro_f1, all_labels, all_preds


def plot_loss_curve(train_losses, val_losses, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_metric_curve(val_accs, val_f1s, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(val_accs, label="Val Accuracy")
    plt.plot(val_f1s, label="Val Macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train():
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_root="cropped_belgiumts",
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        val_ratio=0.2,
        augmented=False,   # baseline 先设 False
        num_workers=2,
    )

    num_classes = len(class_names)
    print("Using classes:", class_names)
    print("Num classes:", num_classes)

    model = SimpleCNN(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []
    val_accs = []
    val_f1s = []

    best_val_f1 = -1
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss, val_acc, val_f1, _, _ = evaluate_model(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "best_simplecnn_belgiumts.pth")
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    plot_loss_curve(train_losses, val_losses, OUTPUT_DIR / "loss_curve.png")
    plot_metric_curve(val_accs, val_f1s, OUTPUT_DIR / "val_metrics.png")

    print("\nLoading best model for final test evaluation...")
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_simplecnn_belgiumts.pth", map_location=DEVICE))

    test_loss, test_acc, test_f1, y_true, y_pred = evaluate_model(model, test_loader, criterion)

    print("\n===== FINAL TEST RESULTS =====")
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test Macro F1 : {test_f1:.4f}")

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
    print("\nClassification Report:\n")
    print(report)

    with open(OUTPUT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        save_path=OUTPUT_DIR / "confusion_matrix.png",
    )

    with open(OUTPUT_DIR / "final_test_results.txt", "w", encoding="utf-8") as f:
        f.write("===== FINAL TEST RESULTS =====\n")
        f.write(f"Test Loss     : {test_loss:.4f}\n")
        f.write(f"Test Accuracy : {test_acc:.4f}\n")
        f.write(f"Test Macro F1 : {test_f1:.4f}\n")

    print(f"\nAll outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    train()