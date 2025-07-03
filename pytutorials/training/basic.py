import torch
import torch.nn as nn
from pytutorials.data.reverse_string import CHAR2IDX, PAD_TOKEN, VOCAB_SIZE

def make_padding_mask(batch_tok: torch.Tensor) -> torch.Tensor:
    """True where PAD so MultiheadAttention can ignore them."""
    return batch_tok.eq(CHAR2IDX[PAD_TOKEN])

def create_optimizer_and_loss(model, lr=1e-3):
    criterion = nn.CrossEntropyLoss(ignore_index=CHAR2IDX[PAD_TOKEN])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return criterion, optimizer

def run_epoch(model, loader, criterion, optimizer=None, train=True):
    model.train(mode=train)
    total, correct = 0, 0

    for inp, tgt in loader:
        mask   = make_padding_mask(inp)
        logits = model(inp, padding_mask=mask)          # (batch, seq, vocab)
        loss   = criterion(logits.view(-1, VOCAB_SIZE), tgt.view(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # crude accuracy (ignoring PAD & SOS)
        with torch.no_grad():
            preds    = logits.argmax(-1)
            valid    = tgt.ne(CHAR2IDX[PAD_TOKEN])
            correct += (preds.eq(tgt) & valid).sum().item()
            total   += valid.sum().item()

    return loss.item(), correct / total
