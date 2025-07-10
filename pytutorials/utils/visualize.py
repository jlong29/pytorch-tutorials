import torch
import matplotlib.pyplot as plt
import numpy as np

def compute_entropy_divergence(attn_maps):
    """Last Layer: Compute mean and std of attention entropies for qualitative tracking."""
    last_layer = attn_maps[-1]  # Shape: [B, H, T, T]
    if last_layer.dim() != 4:
        raise ValueError(f"Expected attention map with 4 dims [B, H, T, T], got {last_layer.shape}")

    p = last_layer.clamp(min=1e-9)  # Avoid log(0)
    entropy = -(p * p.log()).sum(dim=-1)  # [B, H, T]
    entropy = entropy.mean(dim=-1)       # [B, H] — mean over query positions

    return entropy.mean().item(), entropy.std().item()  # mean and std across batch & heads

def visualize_attention_heads(attn_map, sample_idx=0, layer_idx=0, save_path=None):
    """
    Visualize attention maps for all heads of a given layer and sample.
    attn_map: Tensor of shape [B, H, T, T] or [H, T, T]
    """
    if attn_map.ndim == 4:
        # Standard: [B, H, T, T]
        attn = attn_map[sample_idx].detach().cpu().numpy()  # shape [H, T, T]
    elif attn_map.ndim == 3:
        # Possibly already [H, T, T]
        attn = attn_map.detach().cpu().numpy()
    else:
        raise ValueError(f"Expected attention map of dim 3 or 4, got {attn_map.ndim}")

    num_heads = attn.shape[0]
    fig, axes = plt.subplots(1, num_heads, figsize=(3 * num_heads, 3))
    fig.suptitle(f"Layer {layer_idx} Attention Heads for Sample {sample_idx}")

    for h in range(num_heads):
        ax = axes[h] if num_heads > 1 else axes
        ax.imshow(attn[h], cmap='viridis', aspect='auto')
        ax.set_title(f"Head {h}")
        ax.set_xlabel("Key Pos")
        ax.set_ylabel("Query Pos")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_metrics(history, title="", save_path=None):
    keys = [k for k in history if k.startswith("train")]
    fig, axs = plt.subplots(1, len(keys), figsize=(5*len(keys), 4))

    if not isinstance(axs, np.ndarray):
        axs = [axs]

    for i, k in enumerate(keys):
        val_k = k.replace("train", "val")
        axs[i].plot(history[k], label="train")
        axs[i].plot(history[val_k], label="val")
        axs[i].set_title(k.split("_")[1])
        axs[i].legend()
        axs[i].grid(True)

    plt.suptitle(title)
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
