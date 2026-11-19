"""Byte-level dataset: a text file split into train (90%) and validation (10%).

Default corpus: Tiny Shakespeare (about 1.1 MB, public domain), downloaded once.
Any UTF-8 text file can be used instead with --data.
"""
import os
import urllib.request

import numpy as np
import torch

DEFAULT_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_bytes(path, url=DEFAULT_URL):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        urllib.request.urlretrieve(url, path)
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    split = int(0.9 * len(data))
    return torch.from_numpy(data[:split].copy()), torch.from_numpy(data[split:].copy())


class RandomChunks:
    """Samples random [batch, block+1] windows. Each rank uses its own seed, so in
    DDP every GPU sees different samples without a sampler or sharding."""

    def __init__(self, data, block_size, batch_size, device, seed):
        self.data = data
        self.block = block_size
        self.batch = batch_size
        self.device = torch.device(device)
        self.gen = torch.Generator().manual_seed(seed)

    def next(self):
        ix = torch.randint(len(self.data) - self.block - 1, (self.batch,), generator=self.gen)
        chunk = torch.stack([self.data[i:i + self.block + 1] for i in ix.tolist()]).long()
        if self.device.type == "cuda":
            chunk = chunk.pin_memory().to(self.device, non_blocking=True)
        return chunk[:, :-1], chunk[:, 1:]
