import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import time

from data_loader import ByteStreamDataset, create_dummy_data
from model import LiquidStreamModel, ModelArgs

def train():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    data_path = "train_data.txt"
    if not os.path.exists(data_path):
        print("Creating dummy data...")
        # Create a more complex pattern: Sine wave text representation
        import numpy as np
        xx = np.linspace(0, 100, 10000)
        yy = np.sin(xx)
        # Convert to text stream of "val: +0.123\n"
        content = ""
        for y_val in yy:
            content += f"val: {y_val:.3f}\n"

        with open(data_path, 'w') as f:
            f.write(content)

    dataset = ByteStreamDataset(data_path, seq_len=128)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Model
    # Small model for quick MVS
    args = ModelArgs(
        d_model=64,
        n_layer=2,
        vocab_size=256,
        d_state=16,
        dt_rank='auto'
    )
    model = LiquidStreamModel(args).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Train Loop
    epochs = 2
    losses = []

    print("Starting training...")
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x) # (B, L, Vocab)

            # Reshape for loss
            loss = criterion(logits.view(-1, args.vocab_size), y.view(-1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch} Finished | Avg Loss {avg_loss:.4f}")

    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f}s")

    # Verification: Generate
    print("\nGenerative Verification:")
    model.eval()
    start_seq = list(b"val: ")
    input_ids = torch.tensor([start_seq], dtype=torch.long).to(device)

    generated = list(start_seq)

    with torch.no_grad():
        for _ in range(50):
            logits = model(input_ids)
            # Pick next char (greedy or sample)
            # Greedy
            next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            # Keep context window fixed if needed, but Mamba handles full context.
            # Here we just keep appending input for simplicity, but simple inference usually re-injects full state.
            # Since our implementation is naive (re-computes entire history), this gets slower O(L^2) if we don't impl state passing.
            # Efficient Mamba inference passes 'h' state.
            # Our prototype re-runs forward on growing sequence for simplicity.

            generated.append(next_token.item())

    gen_text = bytes(generated).decode('utf-8', errors='replace')
    print(f"Generated: {gen_text}")

    # Verification assert
    if losses[-1] < losses[0]:
        print("SUCCESS: Loss decreased.")
    else:
        print("WARNING: Loss did not decrease significantly.")

if __name__ == "__main__":
    train()
