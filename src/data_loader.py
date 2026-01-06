import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class ByteStreamDataset(Dataset):
    """
    A PyTorch Dataset that serves raw bytes from a file as a continuous stream.
    No tokenization. Just raw 0-255 inputs.
    """
    def __init__(self, file_path, seq_len=1024):
        """
        Args:
            file_path (str): Path to the text/binary file.
            seq_len (int): Length of each sequence chunk.
        """
        self.seq_len = seq_len

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # logical "read" of the file
        with open(file_path, 'rb') as f:
            self.data = np.frombuffer(f.read(), dtype=np.uint8)

        self.total_len = len(self.data)
        # We want to be able to pick any start point essentially, but for a standard dataset
        # we usually chop it into chunks.
        self.n_chunks = self.total_len // seq_len

    def __len__(self):
        # We can effectively return as many chunks as fit
        return self.n_chunks

    def __getitem__(self, idx):
        # Simple chunking strategy
        start = idx * self.seq_len
        end = start + self.seq_len + 1 # +1 for target (next byte)

        # Wrap around or stop? Let's stop/pad if near end, but for now strict chunking
        if end > self.total_len:
            # Padding not implemented for simplicity in this MVP, just cut it shorter or loop?
            # Let's loop for infinite stream feeling, or just take what's left and pad
            # Ideally we just don't access out of bounds since __len__ restricts it.
            chunk = self.data[start:self.total_len]
            # Simple zero pad
            chunk = np.pad(chunk, (0, (self.seq_len + 1) - len(chunk)), 'constant')
        else:
            chunk = self.data[start:end]

        # x is input, y is target (shifted by 1)
        x = torch.from_numpy(chunk[:-1].astype(np.int64))  # (seq_len, )
        y = torch.from_numpy(chunk[1:].astype(np.int64))   # (seq_len, )

        return x, y

def create_dummy_data(path, size=1024*100):
    """
    Creates a dummy text file for testing if one doesn't exist.
    """
    with open(path, 'w') as f:
        # Repeating pattern for easy learning
        f.write("Liquid Analog Stream " * (size // 20))

if __name__ == "__main__":
    # Quick test
    dummy_path = "dummy_stream.txt"
    create_dummy_data(dummy_path)

    dataset = ByteStreamDataset(dummy_path, seq_len=64)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(f"Dataset size: {len(dataset)}")
    for x, y in loader:
        print("Input shape:", x.shape)
        print("Target shape:", y.shape)
        print("Sample input (text):", "".join([chr(c) for c in x[0][:20]]))
        break
