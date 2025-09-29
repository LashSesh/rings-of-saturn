"""Demonstration script showcasing ResonanceLoss on MNIST/CIFAR datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

from .resonance_loss import ResonanceLoss


def _load_dataset(batch_size: int = 64) -> Tuple[DataLoader, DataLoader, int]:
    """Load the MNIST (or CIFAR10 if available) dataset with transforms."""

    try:
        from torchvision import datasets, transforms
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "torchvision is required to run the demo training script."
        ) from exc

    transform = transforms.Compose([transforms.ToTensor()])

    try:
        train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        input_channels = 3
    except Exception:  # pragma: no cover - fallback path
        train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        input_channels = 1

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, input_channels


class ResonanceCNN(nn.Module):
    """Simple convolutional network exposing intermediate features."""

    def __init__(self, input_channels: int, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7 if input_channels == 1 else 64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        feats = self.features(x)
        flat = torch.flatten(feats, start_dim=1)
        logits = self.classifier(flat)
        return logits, flat


@dataclass
class TrainingConfig:
    epochs: int = 1
    lr: float = 1e-3
    batch_size: int = 64


def train(config: TrainingConfig = TrainingConfig()) -> None:
    """Run a lightweight training loop using :class:`ResonanceLoss`."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, input_channels = _load_dataset(config.batch_size)

    model = ResonanceCNN(input_channels=input_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = ResonanceLoss(nn.CrossEntropyLoss())

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            logits, features = model(data)
            loss = criterion(logits, target, features)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 100 == 0:
                avg_loss = running_loss / (batch_idx + 1)
                print(f"Epoch {epoch + 1} Batch {batch_idx}: loss={avg_loss:.4f}")

        validate(model, test_loader, device)


def validate(model: ResonanceCNN, data_loader: DataLoader, device: torch.device) -> None:
    """Evaluate the model and print validation accuracy."""

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            logits, _ = model(data)
            predictions = logits.argmax(dim=1)
            correct += (predictions == target).sum().item()
            total += target.size(0)

    accuracy = 100.0 * correct / max(total, 1)
    print(f"Validation accuracy: {accuracy:.2f}%")


def main() -> None:  # pragma: no cover - script entry point
    train()


if __name__ == "__main__":  # pragma: no cover - script execution guard
    main()

