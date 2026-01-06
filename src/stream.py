"""
stream.py - Liquid Analog Stream Inference

This implements the "Radio Station" pattern where the model is a continuous
signal rather than a request/response chatbot.

Usage:
    python stream.py "Initial context"
"""

import sys
import time
import torch
from model import LiquidStreamModel, ModelArgs

def stream_flow(model, initial_context: bytes, device='cpu'):
    """
    Generator that yields bytes continuously.
    The model "lives" as a continuous signal.
    """
    model.eval()

    # Initialize with context
    tokens = list(initial_context)
    context = torch.tensor([tokens], dtype=torch.long, device=device)

    while True:
        with torch.no_grad():
            logits = model(context)

            # Sample or greedy
            # Using temperature sampling for variety
            probs = torch.softmax(logits[:, -1, :] / 0.8, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Yield the byte
            byte_val = next_token.item()
            yield byte_val

            # Update context (sliding window for efficiency)
            context = torch.cat([context, next_token], dim=1)
            if context.shape[1] > 256:  # Keep context bounded
                context = context[:, -256:]

def main():
    # Load model
    args = ModelArgs(d_model=64, n_layer=2, dt_min=0.001, dt_max=0.1)
    model = LiquidStreamModel(args)

    # Try to load weights if available
    try:
        model.load_state_dict(torch.load('model.pt', map_location='cpu', weights_only=True))
        print("Loaded model weights.", file=sys.stderr)
    except:
        print("Using random weights (no model.pt found).", file=sys.stderr)

    # Initial context
    initial = sys.argv[1].encode('utf-8') if len(sys.argv) > 1 else b"The liquid stream flows"

    print(f"Starting stream with: {initial.decode()}", file=sys.stderr)
    print("=" * 40, file=sys.stderr)

    # Stream output
    for byte_val in stream_flow(model, initial):
        try:
            sys.stdout.buffer.write(bytes([byte_val]))
            sys.stdout.flush()
            time.sleep(0.02)  # 50 bytes/sec for visual effect
        except BrokenPipeError:
            break  # Handle pipe close gracefully

if __name__ == "__main__":
    main()
