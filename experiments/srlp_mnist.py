import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

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
# Per-layer gradient buffers (STEP A1)
# -----------------------------
layer_gradient_buffers = {
    name: deque(maxlen=20)
    for name, _ in model.named_parameters()
}

# -----------------------------
# 4. Loss and optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 5. SRLP setup (global)
# -----------------------------
loss_buffer = deque(maxlen=20)

# -----------------------------
# Logging
# -----------------------------
pressure_history = []
loss_history = []
batch_losses = []
gradient_norms = []

spike_count = 0
SPIKE_WINDOW = 10
SPIKE_THRESHOLD = 0.10

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

        # Loss spike detection
        if len(batch_losses) >= SPIKE_WINDOW:
            recent_mean = np.mean(batch_losses[-SPIKE_WINDOW:])
            if loss.item() > recent_mean + SPIKE_THRESHOLD:
                spike_count += 1

        batch_losses.append(loss.item())
        loss.backward()

        # -------- Per-layer gradient norm tracking (STEP A1) --------
        for name, param in model.named_parameters():
            if param.grad is not None:
                layer_gradient_buffers[name].append(
                    param.grad.data.norm(2).item()
                )
        # ------------------------------------------------------------

        # -------- Global gradient norm tracking --------
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        total_norm = total_norm ** 0.5
        gradient_norms.append(total_norm)
        # ------------------------------------------------

        # -------- SRLP global pressure --------
        loss_buffer.append(loss.item())

        if len(loss_buffer) == loss_buffer.maxlen:
            loss_std = torch.std(torch.tensor(list(loss_buffer))).item()
            k = 5.0
            pressure = 1.0 / (1.0 + k * loss_std)
            pressure = max(0.6, min(1.0, pressure))
        else:
            pressure = 1.0

        for param in model.parameters():
            if param.grad is not None:
                param.grad.mul_(pressure)
        # -------------------------------------

        pressure_history.append(pressure)
        loss_history.append(loss.item())

        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
    print(f"  Sample pressures: {pressure_history[-10:]}")

    loss_std_overall = np.std(batch_losses)
    print(f"Training Loss Std (Volatility): {loss_std_overall:.6f}")

    start_idx = int(len(batch_losses) * 0.50)
    late_std = np.std(batch_losses[start_idx:])
    print(f"Late-stage Loss Std (Volatility): {late_std:.6f}")
    print(f"Loss Spike Count: {spike_count}")

    # Debug: per-layer gradient volatility (STEP A1 verification)
    for name in list(layer_gradient_buffers.keys())[:3]:
        buf = list(layer_gradient_buffers[name])
        if len(buf) > 1:
            print(f"Layer {name} grad-norm std: {np.std(buf):.6f}")
    print()

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

print(f"Gradient Norm Mean: {np.mean(gradient_norms):.6f}")
print(f"Gradient Norm Std: {np.std(gradient_norms):.6f}\n")

# -----------------------------
# 8. Visualization
# -----------------------------
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
