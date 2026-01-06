import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import time
import os
from mlx_model import LiquidStreamModel, ModelArgs

def load_data(path, seq_len=128):
    if not os.path.exists(path):
        xx = np.linspace(0, 100, 10000)
        yy = np.sin(xx)
        content = ""
        for y_val in yy:
            content += f"val: {y_val:.3f}\n"
        with open(path, 'w') as f:
            f.write(content)

    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)

    n_chunks = len(data) // (seq_len + 1)
    data = data[:n_chunks * (seq_len+1)]
    data = data.reshape(-1, seq_len+1)
    X = data[:, :-1].astype(np.int32)
    Y = data[:, 1:].astype(np.int32)

    # Check data statistics
    print(f"Data shape: {X.shape}")
    print(f"Unique bytes: {len(np.unique(Y))}") # Should be 10-20

    return mx.array(X), mx.array(Y)

def loss_fn(model, x, y):
    logits = model(x)
    return mx.mean(nn.losses.cross_entropy(logits, y))

def train():
    data_path = "train_data.txt"
    X, Y = load_data(data_path, seq_len=128)

    batch_size = 8
    n_samples = X.shape[0]

    args = ModelArgs(
        d_model=64,
        n_layer=2,
        vocab_size=256,
        d_state=16
    )
    model = LiquidStreamModel(args)
    mx.eval(model.parameters())

    # Print params (simple recursion)
    def count_params(tree):
        if isinstance(tree, dict):
            return sum(count_params(v) for v in tree.values())
        elif isinstance(tree, list):
            return sum(count_params(v) for v in tree)
        elif hasattr(tree, 'size'):
            return tree.size
        return 0

    total_params = count_params(model.parameters())
    print(f"Total Parameters: {total_params}")

    optimizer = optim.AdamW(learning_rate=1e-3)

    # No compile for debugging
    def step(x, y):
        loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
        optimizer.update(model, grads)
        return loss, grads

    print("Starting MLX training (DEBUG MODE: No Compile)...")
    start_time = time.time()
    epochs = 2

    for epoch in range(epochs):
        perm = mx.array(np.random.permutation(n_samples))
        X = X[perm]
        Y = Y[perm]

        epoch_loss = 0
        steps = 0

        for i in range(0, n_samples, batch_size):
            x_b = X[i:i+batch_size]
            y_b = Y[i:i+batch_size]

            if x_b.shape[0] != batch_size:
                continue

            loss, grads = step(x_b, y_b)
            mx.eval(loss)

            # Check grad norm
            if steps == 0:
                def compute_grad_norm(tree):
                    if isinstance(tree, dict):
                        return sum(compute_grad_norm(v) for v in tree.values())
                    elif isinstance(tree, list):
                        return sum(compute_grad_norm(v) for v in tree)
                    elif hasattr(tree, 'size'):
                        mx.eval(tree)
                        return mx.sum(tree * tree).item()
                    return 0
                grad_norm = compute_grad_norm(grads) ** 0.5
                print(f"Grad Norm: {grad_norm:.4f}")

            epoch_loss += loss.item()
            steps += 1

            if steps % 10 == 0:
                print(f"Epoch {epoch} | Step {steps} | Loss {loss.item():.4f}")

        print(f"Epoch {epoch} Done | Avg Loss {epoch_loss/steps:.4f}")

    print(f"Training finished in {time.time() - start_time:.2f}s")

    # Verification
    print("Generating...")
    start_token = mx.array([ord('v')], dtype=mx.int32).reshape(1, 1)

    curr = start_token
    generated = [ord('v')]

    for _ in range(50):
        logits = model(curr)
        next_tok = mx.argmax(logits[:, -1, :], axis=-1).reshape(1, 1)
        curr = mx.concatenate([curr, next_tok], axis=1)

        mx.eval(next_tok)
        next_val = next_tok[0,0].item()
        generated.append(next_val)

    print("Generated:", bytes(generated).decode('utf-8', errors='replace'))

if __name__ == "__main__":
    train()
