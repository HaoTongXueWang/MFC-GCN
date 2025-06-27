import torch
import torch.nn.functional as F
import numpy as np
import copy
from sklearn import metrics
from torch_geometric.utils import dropout_adj
from torch_geometric.data import Data
from losses import weighted_contrastive_loss, fusion_contrastive_loss, gate_diversity_loss

def identify_hard_samples(logits, labels, mask, ratio):
    probs = torch.sigmoid(logits[mask])
    true_labels = labels[mask].float()
    difficulty = torch.abs(probs - 0.5)
    _, indices = torch.topk(difficulty, k=int(ratio * difficulty.size(0)), largest=False)
    global_indices = torch.where(mask)[0][indices]
    hard_mask = torch.zeros_like(mask, dtype=torch.bool)
    hard_mask[global_indices] = True
    print(f"Hard samples identified: {hard_mask.sum().item()} / {mask.sum().item()} training nodes")
    return hard_mask

def train(model, optimizer, networks, labels, train_mask, epoch, pos_weight):
    model.train()
    optimizer.zero_grad()

    with torch.no_grad():
        _, _, _, logits, _, _ = model(networks, train_mode=False)
        hard_sample_mask = identify_hard_samples(
            logits, labels, train_mask, model.config.hard_sample_ratio
        )

    z_list, proj_list, fusion_proj, logits, gate_weights, moe_loss = model(
        networks, train_mode=True, hard_sample_mask=hard_sample_mask
    )

    cls_loss = F.binary_cross_entropy_with_logits(
        logits[train_mask],
        labels[train_mask].float(),
        pos_weight=pos_weight
    )

    lambda_contrastive = model.config.lambda_contrastive * np.exp(-model.config.lambda_decay_rate * epoch)
    lambda_fusion_contrast = model.config.lambda_fusion_contrast * np.exp(-model.config.lambda_decay_rate * epoch)
    contrast_loss = weighted_contrastive_loss(
        proj_list,
        model.config.temperature,
        train_mask,
        model.config.network_weights,
        hard_sample_mask
    )
    fusion_contrast_loss = fusion_contrastive_loss(
        fusion_proj,
        model.config.temperature,
        train_mask,
        hard_sample_mask
    )

    gate_loss = gate_diversity_loss(gate_weights[train_mask])

    moe_aux_loss = 0
    if moe_loss:
        for loss_name, loss_val in moe_loss.items():
            moe_aux_loss += loss_val

    total_loss = cls_loss + lambda_contrastive * contrast_loss + \
                 lambda_fusion_contrast * fusion_contrast_loss + \
                 0.01 * gate_loss + moe_aux_loss

    total_loss.backward()
    optimizer.step()

    if epoch % model.config.weight_update_freq == 0:
        train_performance = model.evaluate_network_performance(networks, labels, train_mask)
        model.update_network_weights(train_performance, epoch)

    print(f'Epoch {epoch:03d}, Lambda_Con: {lambda_contrastive:.4f}, Lambda_Fus: {lambda_fusion_contrast:.4f}')
    print(f'  Losses - CLS: {cls_loss:.4f}, Contrast: {contrast_loss:.4f}, '
          f'Fusion: {fusion_contrast_loss:.4f}, Gate: {gate_loss:.4f}, MoE: {moe_aux_loss:.4f}')

    with torch.no_grad():
        avg_gates = gate_weights.mean(dim=0).cpu().numpy()
        print(f'  Expert Usage: {", ".join([f"{w:.3f}" for w in avg_gates])}')

    return total_loss.item()

@torch.no_grad()
def evaluate_with_predictions(model, networks, labels, mask):
    model.eval()
    _, _, _, logits, _, _ = model(networks, train_mode=False)
    preds = torch.sigmoid(logits[mask]).cpu().numpy()
    trues = labels[mask].cpu().numpy()
    roc_auc = metrics.roc_auc_score(trues, preds)
    pr_auc = metrics.average_precision_score(trues, preds)
    return preds, trues, roc_auc, pr_auc

def calculate_pos_weight(labels, mask):
    labels = labels[mask].cpu()
    num_pos = torch.sum(labels == 1).float()
    num_neg = torch.sum(labels == 0).float()
    pos_weight = num_neg / (num_pos + 1e-5)
    return torch.tensor(pos_weight, dtype=torch.float32)