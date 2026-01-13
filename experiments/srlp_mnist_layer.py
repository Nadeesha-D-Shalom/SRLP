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
torch.manual_seed(42)
np.random.seed(42)

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
# 5. Metrics + buffers
# -----------------------------
epochs = 3

# Loss tracking
loss_history = []
batch_losses = []

# Spike tracking
spike_count = 0
SPIKE_WINDOW = 10
SPIKE_THRESHOLD = 0.10

# Global gradient norm tracking
gradient_norms = []

# Per-layer gradient volatility buffers (store grad-norm per layer)
LAYER_BUF_LEN = 20
layer_gradnorm_buffers = {
    name: deque(maxlen=LAYER_BUF_LEN)
    for name, _ in model.named_parameters()
}

# Per-layer pressure history (for later plotting/inspection)
layer_pressure_history = {
    name: []
    for name, _ in model.named_parameters()
}

# -----------------------------
# 6. SRLP-L parameters
# -----------------------------
K_LAYER = 5.0        # sensitivity to grad volatility (tune later)
P_MIN = 0.60         # lower bound pressure
P_MAX = 1.00         # upper bound pressure

# -----------------------------
# 7. Training loop
# -----------------------------
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Spike count (loss-based, independent from SRLP)
        if len(batch_losses) >= SPIKE_WINDOW:
            recent_mean = float(np.mean(batch_losses[-SPIKE_WINDOW:]))
            if float(loss.item()) > recent_mean + SPIKE_THRESHOLD:
                spike_count += 1

        batch_losses.append(float(loss.item()))
        loss.backward()

        # -----------------------------
        # A) Measure per-layer grad norms
        # -----------------------------
        per_layer_norm = {}
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            gnorm = float(param.grad.data.norm(2).item())
            per_layer_norm[name] = gnorm
            layer_gradnorm_buffers[name].append(gnorm)

        # -----------------------------
        # B) Compute SRLP-L pressure per layer
        #    pressure_l = clamp(1/(1 + K * std(layer_gradnorm_window)))
        # -----------------------------
        per_layer_pressure = {}
        for name, _ in model.named_parameters():
            buf = layer_gradnorm_buffers[name]
            if len(buf) < buf.maxlen:
                pressure = P_MAX
            else:
                g_std = float(np.std(list(buf)))
                pressure = 1.0 / (1.0 + (K_LAYER * g_std))
                if pressure < P_MIN:
                    pressure = P_MIN
                if pressure > P_MAX:
                    pressure = P_MAX

            per_layer_pressure[name] = float(pressure)
            layer_pressure_history[name].append(float(pressure))

        # -----------------------------
        # C) Apply per-layer pressure
        # -----------------------------
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            param.grad.mul_(per_layer_pressure[name])

        # -----------------------------
        # D) Global gradient norm (after scaling)
        # -----------------------------
        total_norm_sq = 0.0
        for param in model.parameters():
            if param.grad is None:
                continue
            n = float(param.grad.data.norm(2).item())
            total_norm_sq += n * n
        gradient_norms.append(float(total_norm_sq ** 0.5))

        optimizer.step()

        loss_history.append(float(loss.item()))
        total_loss += float(loss.item())

    avg_loss = total_loss / float(len(train_loader))
    loss_std_overall = float(np.std(batch_losses))

    # Late-stage volatility (ignore first 50%)
    WARMUP_RATIO = 0.50
    start_idx = int(len(batch_losses) * WARMUP_RATIO)
    late_std = float(np.std(batch_losses[start_idx:]))

    print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
    print(f"Training Loss Std (Volatility): {loss_std_overall:.6f}")
    print(f"Late-stage Loss Std (Volatility): {late_std:.6f}")
    print(f"Loss Spike Count: {spike_count}")

    # Print 3 key layers pressure sample (last 10 values)
    key_layers = ["net.1.weight", "net.1.bias", "net.3.weight"]
    for lname in key_layers:
        if lname in layer_pressure_history:
            tail = layer_pressure_history[lname][-10:]
            print(f"  {lname} pressure sample: {tail}")
    print("")

# -----------------------------
# 8. Evaluation
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

        total += int(labels.size(0))
        correct += int((predictions == labels).sum().item())

accuracy = 100.0 * float(correct) / float(total)
print(f"Test Accuracy: {accuracy:.2f}%\n")

print(f"Gradient Norm Mean: {float(np.mean(gradient_norms)):.6f}")
print(f"Gradient Norm Std: {float(np.std(gradient_norms)):.6f}\n")

# -----------------------------
# 9. Plots
# -----------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Step")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
# Plot only key layer pressures for readability
for lname in ["net.1.weight", "net.3.weight"]:
    if lname in layer_pressure_history:
        plt.plot(layer_pressure_history[lname], label=lname)

plt.title("Per-Layer Pressure (SRLP-L)")
plt.xlabel("Step")
plt.ylabel("Pressure")
plt.legend()

plt.tight_layout()
plt.show()
