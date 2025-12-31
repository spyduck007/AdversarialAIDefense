import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from models.mnist_cnn import MNISTCNN
from defenses.anomaly_detector import AnomalyDetector
from defenses.robustness_wrapper import RobustnessWrapper
from utils.metrics import (
    plot_confusion_matrix,
    plot_roc_curve,
    calculate_attack_success_rate,
)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def train_model(epochs=5):
    print(f"Training model for {epochs} epochs...")
    device = get_device()
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        root="data/raw", train=True, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = MNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/mnist_cnn.pt")
    print("Model saved to models/mnist_cnn.pt")


def generate_attacks(eps=0.3, attack_type="fgsm"):
    print(
        f"Generating {attack_type.upper()} adversarial examples with epsilon={eps}..."
    )
    device = torch.device("cpu")

    model = MNISTCNN()
    try:
        model.load_state_dict(torch.load("models/mnist_cnn.pt", map_location=device))
    except FileNotFoundError:
        print("Error: Model not found. Please run 'python cli.py train' first.")
        return

    model.eval()

    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST(
        root="data/raw", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    classifier = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=None,
        input_shape=(1, 28, 28),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        device_type="cpu",
    )

    if attack_type == "fgsm":
        attack = FastGradientMethod(estimator=classifier, eps=eps)
    elif attack_type == "pgd":
        attack = ProjectedGradientDescent(
            estimator=classifier, eps=eps, eps_step=0.01, max_iter=40
        )
    else:
        print(f"Error: Unknown attack type '{attack_type}'")
        return

    x_test, y_test = next(iter(test_loader))
    x_test_np = x_test.numpy()

    print("Generating adversarial examples...")
    x_adv = attack.generate(x=x_test_np)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    np.save(f"data/processed/x_adv_{attack_type}.npy", x_adv)
    np.save(f"data/processed/y_adv_{attack_type}.npy", y_test.numpy())
    print(f"Saved {len(x_adv)} adversarial examples to data/processed/")


def evaluate_defense(attack_type="fgsm"):
    print(f"Evaluating defense mechanism against {attack_type.upper()}...")

    try:
        x_adv = np.load(f"data/processed/x_adv_{attack_type}.npy")
        y_adv = np.load(f"data/processed/y_adv_{attack_type}.npy")
    except FileNotFoundError:
        print(
            f"Error: Adversarial data for {attack_type} not found. Please run 'python cli.py attack --type {attack_type}' first."
        )
        return

    transform = transforms.ToTensor()
    clean_dataset = datasets.MNIST(
        root="data/raw", train=False, download=True, transform=transform
    )
    clean_loader = DataLoader(clean_dataset, batch_size=len(x_adv), shuffle=False)
    x_clean, y_clean = next(iter(clean_loader))
    x_clean = x_clean.numpy()

    n = min(len(x_clean), len(x_adv))
    x_clean = x_clean[:n]
    x_adv = x_adv[:n]
    y_clean = y_clean[:n]

    print("\nCalculating Attack Success Rate...")

    device = torch.device("cpu")
    model = MNISTCNN()
    try:
        model.load_state_dict(torch.load("models/mnist_cnn.pt", map_location=device))
        model.eval()
        clean_acc, adv_acc, asr = calculate_attack_success_rate(
            model, x_clean, x_adv, y_clean
        )
        print(f"Clean Accuracy: {clean_acc:.4f}")
        print(f"Adversarial Accuracy: {adv_acc:.4f}")
        print(f"Attack Success Rate: {asr:.4f}")
    except Exception as e:
        print(f"Could not calculate ASR: {e}")

    print(f"\nTraining anomaly detector on {n} clean and {n} adversarial samples...")

    detector = AnomalyDetector()
    detector.fit(x_clean, x_adv)

    x_test = np.concatenate([x_clean, x_adv], axis=0)
    y_test = np.concatenate([np.zeros(n), np.ones(n)])

    y_pred = detector.predict(x_test)

    try:
        features = detector.extract_features(x_test)
        y_scores = detector.pipeline.predict_proba(features)[:, 1]

        print("\nPlotting ROC Curve...")
        roc_auc = plot_roc_curve(
            y_test, y_scores, save_path=f"results/roc_curve_{attack_type}.png"
        )
        print(f"ROC AUC: {roc_auc:.4f}")
    except Exception as e:
        print(f"Could not plot ROC curve: {e}")

    print("\nClassification Report for Anomaly Detector:")
    print(classification_report(y_test, y_pred, target_names=["Clean", "Adversarial"]))

    print("\nPlotting Confusion Matrix...")
    plot_confusion_matrix(
        y_test,
        y_pred,
        classes=["Clean", "Adversarial"],
        save_path=f"results/confusion_matrix_{attack_type}.png",
    )

    Path("defenses").mkdir(exist_ok=True)
    detector.save("defenses/anomaly_detector.joblib")
    print("Anomaly detector saved to defenses/anomaly_detector.joblib")


def main():
    parser = argparse.ArgumentParser(description="Adversarial AI Defense CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    train_parser = subparsers.add_parser("train", help="Train the MNIST model")
    train_parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")

    attack_parser = subparsers.add_parser(
        "attack", help="Generate adversarial examples"
    )
    attack_parser.add_argument(
        "--eps", type=float, default=0.3, help="Epsilon for FGSM/PGD"
    )
    attack_parser.add_argument(
        "--type",
        type=str,
        default="fgsm",
        choices=["fgsm", "pgd"],
        help="Type of attack (fgsm or pgd)",
    )

    defend_parser = subparsers.add_parser("defend", help="Train and evaluate defense")
    defend_parser.add_argument(
        "--type",
        type=str,
        default="fgsm",
        choices=["fgsm", "pgd"],
        help="Type of attack to defend against",
    )

    args = parser.parse_args()

    if args.command == "train":
        train_model(args.epochs)
    elif args.command == "attack":
        generate_attacks(args.eps, args.type)
    elif args.command == "defend":
        evaluate_defense(args.type)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
