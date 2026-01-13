import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import deque
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

# -----------------------------
# 5. SRLP v0 setup
# -----------------------------
loss_buffer = deque(maxlen=20)

LOSS_STD_THRESHOLD = 0.05
PRESSURE_LOW = 0.5
PRESSURE_NORMAL = 1.0

# -----------------------------
# SRLP logging (for visualization)
# -----------------------------
pressure_history = []
loss_history = []

batch_losses = []
# -----------------------------
# 6. Training loop
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
        batch_losses.append(loss.item())
        loss.backward()

        # -------- SRLP v0 logic --------
        loss_buffer.append(loss.item())

        if len(loss_buffer) == loss_buffer.maxlen:
            loss_std = torch.std(torch.tensor(list(loss_buffer))).item()

            # Smooth SRLP v1 pressure
            k = 10.0
            pressure = 1.0 / (1.0 + k * loss_std)

            # Clamp pressure range
            pressure = max(0.5, min(1.0, pressure))
        else:
            pressure = 1.0

        for param in model.parameters():
            if param.grad is not None:
                param.grad.mul_(pressure)
        # --------------------------------

        pressure_history.append(pressure)
        loss_history.append(loss.item())

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
    print(f"  Sample pressures: {pressure_history[-10:]}")

    loss_std_overall = np.std(batch_losses)
    print(f"Training Loss Std (Volatility): {loss_std_overall:.6f}")

    # Late-stage volatility: ignore early training noise
    WARMUP_RATIO = 0.50  # ignore first 50% of batches
    start_idx = int(len(batch_losses) * WARMUP_RATIO)

    late_losses = batch_losses[start_idx:]
    late_std = np.std(late_losses)

    print(f"Late-stage Loss Std (Volatility): {late_std:.6f}\n")

# -----------------------------
# 7. Evaluation
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


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Step")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(pressure_history)
plt.title("Learning Pressure (SRLP)")
plt.xlabel("Step")
plt.ylabel("Pressure")

plt.tight_layout()
plt.show()

