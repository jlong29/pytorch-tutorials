# pytutorials/data/reverse_string.py

import random
import string
import torch
from torch.utils.data import Dataset, DataLoader

# Constants
ALPHABET   = string.ascii_lowercase
PAD_TOKEN  = "<pad>"
SOS_TOKEN  = "<s>"
ALL_TOKENS = [PAD_TOKEN, SOS_TOKEN] + list(ALPHABET)
VOCAB_SIZE = len(ALL_TOKENS)
CHAR2IDX   = {ch: i for i, ch in enumerate(ALL_TOKENS)}
IDX2CHAR   = {i: ch for ch, i in CHAR2IDX.items()}
MAX_LEN    = 16

def encode(seq: str) -> list[int]:
    """String → list of indices incl. <s> at front, padded to MAX_LEN+1."""
    seq = seq.lower()
    assert len(seq) <= MAX_LEN, "too long"
    idxs = [CHAR2IDX[SOS_TOKEN]] + [CHAR2IDX[c] for c in seq]
    idxs += [CHAR2IDX[PAD_TOKEN]] * ((MAX_LEN + 1) - len(idxs))
    return idxs

def decode(idxs: list[int]) -> str:
    """Drop SOS & PAD and turn back into string."""
    return "".join(IDX2CHAR[i] for i in idxs if i > 1)

class ReverseDataset(Dataset):
    """Generate random strings on-the-fly; split = 'train' or 'val'."""
    def __init__(self, split="train", n_samples=10_000):
        random.seed(0 if split=="train" else 1)
        self.samples = [
            "".join(random.choices(ALPHABET, k=random.randint(3, MAX_LEN)))
            for _ in range(n_samples)
        ]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        inp = torch.tensor(encode(s), dtype=torch.long)
        tgt = torch.tensor(encode(s[::-1]), dtype=torch.long)
        return inp, tgt

def get_dataloaders(batch_size=64):
    train_loader = DataLoader(ReverseDataset("train"), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(ReverseDataset("val", n_samples=2000), batch_size=batch_size)
    return train_loader, val_loader
