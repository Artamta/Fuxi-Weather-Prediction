import torch
import torch.nn as nn
import torch.optim as optim
from fuxi import FuXiModel

# ---- Config ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
IN_CHANNELS = 70
OUT_CHANNELS = 70
TIME_STEPS = 2
HEIGHT = 720
WIDTH = 1440
EPOCHS = 3
LEARNING_RATE = 1e-3

# ---- Synthetic Data ----
def get_synthetic_batch(batch_size=BATCH_SIZE):
    # Input: (B, TIME_STEPS, IN_CHANNELS, HEIGHT, WIDTH)
    x = torch.randn(batch_size, TIME_STEPS, IN_CHANNELS, HEIGHT, WIDTH, device=device)
    # Target: (B, OUT_CHANNELS, HEIGHT, WIDTH)
    y = torch.randn(batch_size, OUT_CHANNELS, HEIGHT, WIDTH, device=device)
    return x, y

# ---- Model ----
model = FuXiModel(
    in_channels=IN_CHANNELS,
    out_channels=OUT_CHANNELS,
    embed_dim=192,
    num_heads=2,
    depth=6
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---- Training Loop ----
for epoch in range(EPOCHS):
    model.train()
    x, y = get_synthetic_batch()
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f}")

print("Synthetic training complete!")