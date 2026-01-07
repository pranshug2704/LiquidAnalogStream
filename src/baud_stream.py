"""
baud_stream.py - Constant Baud-Rate Streaming

Outputs bytes at a fixed rate (e.g., 9600 bps) regardless of computation time.
This makes the model behave like a physical signal/radio transmission.
"""

import sys
import time
import threading
import queue
import torch
from model import LiquidStreamModel, ModelArgs

class BaudRateStreamer:
    """
    Outputs bytes at constant baud rate.
    Computation happens in background thread.
    """

    def __init__(self, baud_rate: int = 9600):
        self.baud_rate = baud_rate
        self.byte_delay = 10.0 / baud_rate  # 10 bits per byte (8 data + start + stop)
        self.buffer = queue.Queue(maxsize=256)
        self.running = False
        self.output_thread = None

    def _output_worker(self):
        """Output bytes at constant rate."""
        while self.running or not self.buffer.empty():
            try:
                byte_val = self.buffer.get(timeout=0.1)
                sys.stdout.buffer.write(bytes([byte_val]))
                sys.stdout.flush()
                time.sleep(self.byte_delay)
            except queue.Empty:
                continue

    def start(self):
        """Start the output thread."""
        self.running = True
        self.output_thread = threading.Thread(target=self._output_worker, daemon=True)
        self.output_thread.start()

    def stop(self):
        """Stop and wait for buffer to drain."""
        self.running = False
        if self.output_thread:
            self.output_thread.join(timeout=5.0)

    def push(self, byte_val: int):
        """Push a byte to the output buffer."""
        self.buffer.put(byte_val, block=True)

def stream_with_baud(model, initial: bytes, baud_rate: int = 9600, max_bytes: int = 500):
    """Stream model output at constant baud rate."""
    model.eval()
    streamer = BaudRateStreamer(baud_rate)

    # Initialize
    tokens = list(initial)
    context = torch.tensor([tokens], dtype=torch.long)

    # Push initial context
    for b in initial:
        streamer.push(b)

    streamer.start()

    print(f"\n[Streaming at {baud_rate} baud]", file=sys.stderr)

    try:
        with torch.no_grad():
            for i in range(max_bytes):
                logits = model(context)
                probs = torch.softmax(logits[:, -1, :] / 0.8, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                byte_val = next_token.item()
                streamer.push(byte_val)

                context = torch.cat([context, next_token], dim=1)
                if context.shape[1] > 256:
                    context = context[:, -256:]
    except KeyboardInterrupt:
        pass

    streamer.stop()
    print("\n[Stream ended]", file=sys.stderr)

def main():
    # Setup
    args = ModelArgs(d_model=64, n_layer=2)
    model = LiquidStreamModel(args)

    try:
        model.load_state_dict(torch.load('model.pt', map_location='cpu', weights_only=True))
        print("Loaded model weights", file=sys.stderr)
    except:
        print("Using random weights", file=sys.stderr)

    # Parse args
    baud = int(sys.argv[1]) if len(sys.argv) > 1 else 9600
    initial = (sys.argv[2] if len(sys.argv) > 2 else "val: ").encode('utf-8')

    stream_with_baud(model, initial, baud_rate=baud)

if __name__ == "__main__":
    main()
