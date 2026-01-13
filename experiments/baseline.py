import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np


# -----------------------------
# 1. Basic setup
# -----------------------------
device = torch.device("cpu")

# -----------------------------
# 2. Dataset (MNIST)
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# -----------------------------
# 3. Simple neural network
# -----------------------------
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleMLP().to(device)

# -----------------------------
# 4. Loss and optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


batch_losses = []
gradient_norms = []


spike_count = 0
SPIKE_WINDOW = 10
SPIKE_THRESHOLD = 0.10  # adjust later if needed


# -----------------------------
# 5. Training loop
# -----------------------------
epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        if len(batch_losses) >= SPIKE_WINDOW:
            recent_mean = np.mean(batch_losses[-SPIKE_WINDOW:])
            if loss.item() > recent_mean + SPIKE_THRESHOLD:
                spike_count += 1

        batch_losses.append(loss.item())
        loss.backward()
        # -------- Gradient Norm Measurement --------
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        total_norm = total_norm ** 0.5
        gradient_norms.append(total_norm)
        # ------------------------------------------

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    loss_std_overall = np.std(batch_losses)
    print(f"Training Loss Std (Volatility): {loss_std_overall:.6f}")

    # Late-stage volatility: ignore early training noise
    WARMUP_RATIO = 0.50  # ignore first 50% of batches
    start_idx = int(len(batch_losses) * WARMUP_RATIO)

    late_losses = batch_losses[start_idx:]
    late_std = np.std(late_losses)

    print(f"Late-stage Loss Std (Volatility): {late_std:.6f}")
    print(f"Loss Spike Count: {spike_count}\n")

# -----------------------------
# 6. Simple evaluation
# -----------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

accuracy = 100.0 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%\n")

print(f"Gradient Norm Std: {np.std(gradient_norms):.6f}")
print(f"Gradient Norm Mean: {np.mean(gradient_norms):.6f}\n")
