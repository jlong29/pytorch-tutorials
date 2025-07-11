import torch
import torch.nn as nn
import torch.nn.functional as F
from pytutorials.models.components import PositionalEncoding, TransformerBlock
from pytutorials.data.synthetic_triplets import NUM_CLASSES, TOKEN_TO_ID, ID_TO_TOKEN

# ---- Baseline Transformer ----
class BaselineTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, ff_dim, dropout):
        super().__init__()
        self.nhead        = nhead
        self.embedding    = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, 500)
        self.layers       = nn.ModuleList([
            TransformerBlock(d_model, nhead, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(d_model, NUM_CLASSES)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)

        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x)
            attn_maps.append(attn)
        
        logits = self.classifier(x)
        return logits, attn_maps

# ---- InfoMax Transformer ----
class InfoMaxTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, ff_dim, dropout):
        super().__init__()
        self.embedding    = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, 500)
        self.layers       = nn.ModuleList([
            TransformerBlock(d_model, nhead, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(d_model, NUM_CLASSES)
        self.nheads=nhead

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        attn_maps     = []
        layer_outputs = []
        for i, layer in enumerate(self.layers):
            x, attn = layer(x)
            attn_maps.append(attn)
            
            if i == len(self.layers) - 1:
                layer_outputs.append(x)
        
        logits = self.classifier(x)
        return logits, attn_maps, layer_outputs

    def compute_orthogonality_loss(self, head_output):
        """
        Penalize similarity between per-head output representations.
        head_output: Tensor of shape [B, T, D] where D = H * d_h
        """

        # Split into heads
        B, T, D   = head_output.shape
        H         = self.nheads
        d_h       = D // H
        head_repr = head_output.view(B, T, H, d_h).transpose(1, 2)
        
        # Mean over tokens
        H_out     = head_repr.mean(dim=2)                      # [B, H, d_h]
        H_out     = F.normalize(H_out, dim=-1)                 # unit length per head
        gram      = torch.einsum('bhd,bkd->bhk', H_out, H_out) # [B, H, H]
        identity  = torch.eye(H, device=gram.device)[None]     # [1, H, H]
        return ((gram - identity) ** 2).mean()                 # scalar

    def compute_auxiliary_losses(self, attn_weights, head_outputs, entropy_weight=1.0, orthogonality_weight=1.0):
        # Entropy of attention weights
        def softmax_entropy(attn):
            p = attn.clamp(min=1e-9)
            return -(p * p.log()).sum(dim=-1).mean()

        entropy_loss       = 0
        orthogonality_loss = 0

        if attn_weights:
            for A in attn_weights[-1]:  # last layer only
                entropy_loss += softmax_entropy(A)
            entropy_loss = -entropy_loss / len(attn_weights[-1])  # maximize entropy

        if head_outputs:
            H                  = head_outputs[0]  # shape: [B, T, D]
            orthogonality_loss = self.compute_orthogonality_loss(H)

        return entropy_weight * entropy_loss + orthogonality_weight * orthogonality_loss
