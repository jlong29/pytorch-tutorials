import torch
import torch.nn as nn
from pytutorials.models.components import PositionalEncoding

class MiniTransformerLayer(nn.Module):
    """
    Post-norm variant:
        y = LN(x + MHA(LN(x)))
        z = LN(y + FFN(LN(y)))
    """
    def __init__(self, d_model=128, num_heads=4, d_ff=256, p_drop=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=p_drop, batch_first=True)
        self.ff  = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x, padding_mask=None):
        # --- block 1: self-attention residual ---
        x_norm      = self.ln1(x)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm, key_padding_mask=padding_mask)
        x           = x + self.drop(attn_out)

        # --- block 2: feed-forward residual ---
        y_norm = self.ln2(x)
        ff_out = self.ff(y_norm)
        x      = x + self.drop(ff_out)
        return x

class TinyTransformer(nn.Module):
    def __init__(self,
                 vocab_size,
                 d_model=128,
                 n_layers=2,
                 num_heads=4,
                 d_ff=256,
                 max_len=17):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model)
        self.pos    = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList(
            [MiniTransformerLayer(d_model, num_heads, d_ff) for _ in range(n_layers)]
        )
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, tok, padding_mask=None):
        """
        tok : (batch, seq) integers
        padding_mask : (batch, seq) True on PAD
        """
        x = self.embed(tok)                 # (batch, seq, d_model)
        x = self.pos(x.transpose(0,1))      # → (seq, batch, d_model)
        x = x.transpose(0,1)                # back to (batch, seq, d_model)

        for layer in self.layers:
            x = layer(x, padding_mask)

        return self.proj(x)                 # logits (batch, seq, vocab)
