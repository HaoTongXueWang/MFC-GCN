import torch
import torch.nn.functional as F

def weighted_contrastive_loss(proj_list, temperature, mask, weights, hard_sample_mask=None):
    loss = 0
    n_views = len(proj_list)
    total_weight = 0
    proj_list = [proj[mask] for proj in proj_list]
    sample_weights = torch.ones(proj_list[0].size(0), device=proj_list[0].device)
    if hard_sample_mask is not None:
        sample_weights[hard_sample_mask[mask]] = 2.0

    for i in range(n_views):
        for j in range(i + 1, n_views):
            pair_weight = (weights[i] * weights[j]) ** 0.5
            total_weight += pair_weight
            z1 = F.normalize(proj_list[i], dim=1)
            z2 = F.normalize(proj_list[j], dim=1)
            sim_matrix = torch.mm(z1, z2.t()) / temperature
            pos_sim = torch.diag(sim_matrix)
            loss_ij = -torch.log(torch.exp(pos_sim) / torch.exp(sim_matrix).sum(dim=1))
            loss_ij = loss_ij * sample_weights
            loss += pair_weight * loss_ij.mean()
    return loss / max(total_weight, 1e-8)

def fusion_contrastive_loss(fusion_proj, temperature, mask, hard_sample_mask=None):
    z = F.normalize(fusion_proj[mask], dim=1)
    sim_matrix = torch.mm(z, z.t()) / temperature
    pos_sim = torch.diag(sim_matrix)
    sample_weights = torch.ones(z.size(0), device=z.device)
    if hard_sample_mask is not None:
        sample_weights[hard_sample_mask[mask]] = 2.0
    loss = -torch.log(torch.exp(pos_sim) / torch.exp(sim_matrix).sum(dim=1))
    loss = loss * sample_weights
    return loss.mean()

def gate_diversity_loss(gate_weights, eps=1e-6):
    entropy = -torch.sum(gate_weights * torch.log(gate_weights + eps), dim=1)
    return -entropy.mean()