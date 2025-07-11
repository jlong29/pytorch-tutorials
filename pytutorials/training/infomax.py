# Define the training logic for infomax transformers
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from collections import defaultdict
from pathlib import Path

from pytutorials.data.synthetic_triplets import SyntheticDisentangleDataset, TOKEN_TO_ID, decode_triplets, decode_target_color
from pytutorials.models.infomax import BaselineTransformer, InfoMaxTransformer
from pytutorials.utils.visualize import compute_entropy_divergence, visualize_attention_heads, plot_metrics

# ---- Training Loop ----
def train_model(model, optimizer, criterion, dataloader, model_name="baseline", cur_epoch=0, num_epochs=30, device=None):
    model.train()
    all_loss, all_entropy, all_entropy_std, all_ortho_loss = [], [], [], []

    correct, total = 0, 0

    λ_entropy_base = 1.0
    λ_ortho_base   = 1000.0  # assuming ortho losses are ~1e-4 in scale

    # Schedules
    λ_entropy = λ_entropy_base * (cur_epoch / num_epochs)     # fade-in
    λ_ortho   = λ_ortho_base   * (cur_epoch / num_epochs)     # fade-in

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if model_name == "infomax":
            output, attn_maps, layer_outputs = model(x)
            color_logits  = output[:, 0::3, :]
            loss_main     = criterion(color_logits.reshape(-1, color_logits.size(-1)), y.view(-1))
            aux_loss      = model.compute_auxiliary_losses(attn_maps, layer_outputs,
                                                            entropy_weight=λ_entropy,
                                                            orthogonality_weight=λ_ortho)
            loss          = loss_main + aux_loss
            mean_H, std_H = compute_entropy_divergence(attn_maps)
            ortho_loss    = model.compute_orthogonality_loss(layer_outputs[0])
        else:
            output, attn_maps = model(x)
            color_logits  = output[:, 0::3, :]
            loss          = criterion(color_logits.reshape(-1, color_logits.size(-1)), y.view(-1))
            mean_H, std_H = compute_entropy_divergence(attn_maps)
            ortho_loss = 0.0

        loss.backward()
        optimizer.step()

        all_loss.append(loss.item())
        all_entropy.append(mean_H)
        all_entropy_std.append(std_H)
        all_ortho_loss.append(ortho_loss if isinstance(ortho_loss, float) else ortho_loss.item())

        preds    = color_logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total   += y.numel()

    return {
        "loss": np.mean(all_loss),
        "acc": correct / total,
        "entropy_mean": np.mean(all_entropy),
        "entropy_std": np.mean(all_entropy_std),
        "ortho": np.mean(all_ortho_loss),
    }

# ---- Validation Loop ----
def eval_model(model, criterion, dataloader, model_name="baseline", cur_epoch=0, num_epochs=30, device=None):
    model.eval()
    all_loss, all_entropy, all_entropy_std, all_ortho_loss = [], [], [], []

    correct, total = 0, 0

    λ_entropy_base = 1.0
    λ_ortho_base   = 1000.0

    λ_entropy = λ_entropy_base * (cur_epoch / num_epochs)
    λ_ortho   = λ_ortho_base   * (cur_epoch / num_epochs)

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            if model_name == "infomax":
                output, attn_maps, layer_outputs = model(x)
                color_logits = output[:, 0::3, :]
                loss_main    = criterion(color_logits.reshape(-1, color_logits.size(-1)), y.view(-1))
                aux_loss     = model.compute_auxiliary_losses(attn_maps, layer_outputs,
                                                                entropy_weight=λ_entropy,
                                                                orthogonality_weight=λ_ortho)
                loss         = loss_main + aux_loss
                ortho_loss   = model.compute_orthogonality_loss(layer_outputs[0])
            else:
                output, attn_maps = model(x)
                color_logits = output[:, 0::3, :]
                loss         = criterion(color_logits.reshape(-1, color_logits.size(-1)), y.view(-1))
                ortho_loss   = 0.0

            mean_H, std_H = compute_entropy_divergence(attn_maps)
            all_loss.append(loss.item())
            all_entropy.append(mean_H)
            all_entropy_std.append(std_H)
            all_ortho_loss.append(ortho_loss if isinstance(ortho_loss, float) else ortho_loss.item())

            preds = color_logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.numel()

    return {
        "loss": np.mean(all_loss),
        "acc": correct / total,
        "entropy_mean": np.mean(all_entropy),
        "entropy_std": np.mean(all_entropy_std),
        "ortho": np.mean(all_ortho_loss),
    }

# ---- Main Training Routine ----
def run_experiment(d_model=256, nhead=4, num_layers=3, ff_dim=1024, dropout=0.1, num_epochs=30):
    
    vocab_size   = len(TOKEN_TO_ID)
    train_loader = DataLoader(SyntheticDisentangleDataset(5000), batch_size=64, shuffle=True)
    val_loader   = DataLoader(SyntheticDisentangleDataset(1000), batch_size=64)

    baseline     = BaselineTransformer(vocab_size, d_model, nhead, num_layers, ff_dim, dropout)
    infomax      = InfoMaxTransformer(vocab_size, d_model, nhead, num_layers, ff_dim, dropout)

    opt_base     = torch.optim.Adam(baseline.parameters(), lr=1e-4)
    opt_info     = torch.optim.Adam(infomax.parameters(), lr=1e-4)
    criterion    = nn.CrossEntropyLoss()

    history_base = defaultdict(list)
    history_info = defaultdict(list)
    SAVE_PATH    = Path("./attention_maps")
    SAVE_PATH.mkdir(exist_ok=True)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        tr_base  = train_model(baseline, opt_base, criterion, train_loader, model_name="baseline", cur_epoch=epoch)
        val_base = eval_model(baseline, criterion, val_loader, model_name="baseline", cur_epoch=epoch)
        
        for k, v in tr_base.items(): history_base[f"train_{k}"].append(v)
        for k, v in val_base.items(): history_base[f"val_{k}"].append(v)

        tr_info  = train_model(infomax, opt_info, criterion, train_loader, model_name="infomax", cur_epoch=epoch, num_epochs=num_epochs)
        val_info = eval_model(infomax, criterion, val_loader, model_name="infomax", cur_epoch=epoch, num_epochs=num_epochs)
        
        for k, v in tr_info.items(): history_info[f"train_{k}"].append(v)
        for k, v in val_info.items(): history_info[f"val_{k}"].append(v)

        print("[BASELINE] Loss: {:.4f}  Acc: {:.2%}  Ortho: {:.6f}  Entropy: {:.3f} ± {:.3f}".format(
            val_base['loss'], val_base['acc'], val_base['ortho'], val_base['entropy_mean'], val_base['entropy_std']))
        print("[INFOMAX ] Loss: {:.4f}  Acc: {:.2%}  Ortho: {:.6f}  Entropy: {:.3f} ± {:.3f}".format(
            val_info['loss'], val_info['acc'], val_info['ortho'], val_info['entropy_mean'], val_info['entropy_std']))

        # Qualitative Sample
        sample, target = next(iter(val_loader))
        sample, target = sample.to(device), target.to(device)

        print("RAW input:", sample[0].tolist())
        print("TRIPLETS:", decode_triplets(sample[0].tolist()))

        base_logits = baseline(sample)[0]
        info_logits = infomax(sample)[0]
        
        base_pred   = base_logits[:, 0::3, :].argmax(-1)[0].tolist()
        info_pred   = info_logits[:, 0::3, :].argmax(-1)[0].tolist()

        print(" target  :", decode_target_color(target[0].tolist()))
        print(" baseline:", decode_target_color(base_pred))
        print(" infomax :", decode_target_color(info_pred))

        # Visualization
        _, infomax_attn, _ = infomax(sample)
        visualize_attention_heads(infomax_attn[-1], sample_idx=0, layer_idx=num_layers - 1, save_path=SAVE_PATH/f"infomax_attn_epoch{epoch+1}.png")

        if epoch == num_epochs - 1:
            plot_metrics(history_base, title="Baseline Transformer", save_path=SAVE_PATH/f"baseline_epoch{epoch+1}.png")
            plot_metrics(history_info, title="InfoMax Transformer", save_path=SAVE_PATH/f"infomax_epoch{epoch+1}.png")

    return history_base, history_info
