import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Any

from core.model import MNISTNet


def train(
    model: MNISTNet,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Train the model locally.

    Args:
        model: Model to train.
        train_loader: Client training DataLoader.
        epochs: Number of local training epochs.
        learning_rate: SGD learning rate.
        device: Device used for training.

    Returns:
        Dictionary containing average training loss and accuracy.
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
    )

    model.train()
    model.to(device)

    total_loss = 0.0
    correct = 0
    total = 0

    for _ in range(epochs):
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Clear previous gradients.
            optimizer.zero_grad()

            # Forward pass.
            outputs = model(inputs)

            # Calculate loss.
            loss = criterion(outputs, labels)

            # Backward pass.
            loss.backward()

            # Update model parameters.
            optimizer.step()

            # Accumulate metrics.
            total_loss += loss.item() * inputs.size(0)

            predicted = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
        }

    average_loss = total_loss / total
    accuracy = correct / total

    return {
        "loss": average_loss,
        "accuracy": accuracy,
    }


def evaluate(
    model: MNISTNet,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model on a test set.

    Args:
        model: Model to evaluate.
        test_loader: Evaluation DataLoader.
        device: Device used for evaluation.

    Returns:
        Tuple containing:
            - average loss
            - accuracy
    """

    criterion = nn.CrossEntropyLoss()

    model.eval()
    model.to(device)

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass.
            outputs = model(inputs)

            # Calculate loss.
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)

            predicted = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        return 0.0, 0.0

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy