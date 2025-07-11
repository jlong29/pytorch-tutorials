import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 17):
        super().__init__()
        pe          = torch.zeros(max_len, d_model)
        position    = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term    = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, d_model)
        returns x + P (adds positional encoding)
        """
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x

# ---- Transformer Block with Pre-Norm ----
class TransformerBlock(nn.Module):
    def __init__(self, d_model, heads, ff_dim, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, dropout=dropout, batch_first=True)
        self.ln1  = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model),
        )
        self.ln2     = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        attn_input = self.ln1(x)
        attn_output, attn_weights = self.attn(
            attn_input, attn_input, attn_input,
            need_weights=True,
            average_attn_weights=False,
            attn_mask=attn_mask
        )

        # Residual Pre-norm
        x        = x + self.dropout(attn_output)
        ff_input = self.ln2(x)
        x        = x + self.dropout(self.ff(ff_input))
        return x, attn_weights
