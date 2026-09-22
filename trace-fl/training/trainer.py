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
    """Train the model locally."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    model.train()
    model.to(device)
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
    }

def test(
    model: MNISTNet,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the model on a test set."""
    criterion = nn.CrossEntropyLoss()
    
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy
