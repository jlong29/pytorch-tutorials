import torch
from torch.utils.data import Dataset
import numpy as np

# Global constants
VOCAB_SIZE  = 64
SEQ_LEN     = 30
NUM_CLASSES = 10

# ---- Vocabulary Generation ----
def generate_vocab():
    return {
        'colors': [f"color{i}" for i in range(10)],
        'shapes': [f"shape{i}" for i in range(10)],
        'positions': [f"pos{i}" for i in range(10)]
    }

VOCAB       = generate_vocab()
TOKEN_TO_ID = {}
index       = 0
for group in VOCAB.values():
    for tok in group:
        TOKEN_TO_ID[tok] = index
        index += 1
ID_TO_TOKEN = {i: tok for tok, i in TOKEN_TO_ID.items()}

# ---- Dataset ----
class SyntheticDisentangleDataset(Dataset):
    def __init__(self, size):
        self.size  = size
        self.vocab = VOCAB
        self.data  = []
        for _ in range(size):
            tokens = []
            target = []
            for _ in range(SEQ_LEN):
                color = np.random.choice(self.vocab['colors'])
                shape = np.random.choice(self.vocab['shapes'])
                pos   = np.random.choice(self.vocab['positions'])
                tokens.append((color, shape, pos))
                target.append(TOKEN_TO_ID[color])  # reverse color prediction task
            self.data.append((tokens, target[::-1]))

    def __len__(self): return self.size

    def __getitem__(self, idx):
        tokens, target = self.data[idx]
        input_ids = []
        for color, shape, pos in tokens:
            input_ids.append(TOKEN_TO_ID[color])
            input_ids.append(TOKEN_TO_ID[shape])
            input_ids.append(TOKEN_TO_ID[pos])
        return torch.tensor(input_ids), torch.tensor(target)

# ---- Decoding Utilities ----
def decode(indices):
    return " ".join(ID_TO_TOKEN.get(i, "<UNK>") for i in indices)

def decode_triplets(indices):
    triplets = [(indices[i], indices[i+1], indices[i+2]) for i in range(0, len(indices), 3)]
    return [tuple(ID_TO_TOKEN.get(idx, "<UNK>") for idx in triplet) for triplet in triplets]

def decode_target_color(indices):
    return [ID_TO_TOKEN.get(i, "<UNK>") for i in indices]
