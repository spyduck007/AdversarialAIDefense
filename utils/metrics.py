import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
import seaborn as sns
from pathlib import Path


def plot_confusion_matrix(
    y_true, y_pred, classes, save_path="results/confusion_matrix.png"
):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")
    return cm


def plot_roc_curve(y_true, y_scores, save_path="results/roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"ROC curve saved to {save_path}")
    return roc_auc


def calculate_attack_success_rate(model, x_clean, x_adv, y_true):
    if hasattr(model, "predict"):
        y_pred_clean = np.argmax(model.predict(x_clean), axis=1)
        y_pred_adv = np.argmax(model.predict(x_adv), axis=1)
    else:
        import torch

        device = next(model.parameters()).device
        with torch.no_grad():
            if isinstance(x_clean, np.ndarray):
                x_clean_t = torch.from_numpy(x_clean).to(device).float()
                x_adv_t = torch.from_numpy(x_adv).to(device).float()
            else:
                x_clean_t = x_clean.to(device)
                x_adv_t = x_adv.to(device)

            y_pred_clean = model(x_clean_t).argmax(dim=1).cpu().numpy()
            y_pred_adv = model(x_adv_t).argmax(dim=1).cpu().numpy()

    if not isinstance(y_true, np.ndarray):
        y_true = y_true.numpy()

    clean_acc = accuracy_score(y_true, y_pred_clean)

    adv_acc = accuracy_score(y_true, y_pred_adv)

    correct_clean_mask = y_pred_clean == y_true
    if np.sum(correct_clean_mask) > 0:
        successful_attacks = (
            y_pred_adv[correct_clean_mask] != y_true[correct_clean_mask]
        )
        asr = np.mean(successful_attacks)
    else:
        asr = 0.0

    return clean_acc, adv_acc, asr
