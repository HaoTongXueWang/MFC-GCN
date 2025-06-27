import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, GCNConv
from torch_geometric.utils import dropout_adj  # 添加这行导入
import copy
from sklearn import metrics

class EnhancedGCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, config):
        super().__init__()
        self.conv1 = ChebConv(in_channels, hidden_channels, K=2, normalization="sym")
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(p=config.dropout_rate)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // config.projection_hidden_ratio),
            nn.BatchNorm1d(hidden_dim // config.projection_hidden_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim // config.projection_hidden_ratio, out_dim)
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=config.xavier_gain)

    def forward(self, x):
        return self.net(x)

class FeatureInteraction(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)
        )

    def forward(self, x):
        return self.mlp(x) + x

class MoEGating(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, dropout=0.2, noise_level=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.noise_level = noise_level

        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts)
        )

        nn.init.normal_(self.gate_net[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.gate_net[-1].bias)

    def forward(self, x, training=True):
        gate_logits = self.gate_net(x)

        if training and self.noise_level > 0:
            noise = torch.randn_like(gate_logits) * self.noise_level
            gate_logits = gate_logits + noise

        gate_weights = F.softmax(gate_logits, dim=-1)
        router_z_loss = torch.logsumexp(gate_logits, dim=-1).mean()
        mean_prob_per_expert = gate_weights.mean(dim=0)
        target_prob = torch.ones_like(mean_prob_per_expert) / self.num_experts
        load_balancing_loss = torch.sum(mean_prob_per_expert * torch.log(mean_prob_per_expert / target_prob))

        return gate_weights, router_z_loss, load_balancing_loss

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        return self.out_proj(attn_output)

class MultiNetworkMoEModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoders = nn.ModuleList([
            EnhancedGCNEncoder(config.in_channels, config.hidden_channels, config.out_channels, config)
            for _ in range(5)
        ])

        self.projectors = nn.ModuleList([
            ProjectionHead(config.out_channels, config.hidden_channels, config.projection_size, config)
            for _ in range(5)
        ])

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.out_channels, config.hidden_channels),
                nn.LayerNorm(config.hidden_channels),
                nn.GELU(),
                nn.Linear(config.hidden_channels, config.hidden_channels)
            ) for _ in range(config.num_experts)
        ])

        self.gating = MoEGating(
            input_dim=config.out_channels * 5,
            hidden_dim=config.moe_hidden_dim,
            num_experts=config.num_experts,
            dropout=config.gating_dropout,
            noise_level=config.gate_noise
        )

        self.cross_attention = MultiheadAttention(
            embed_dim=config.out_channels,
            num_heads=config.num_heads,
            dropout=config.attention_dropout
        )

        self.layer_norm1 = nn.LayerNorm(config.out_channels)
        self.layer_norm2 = nn.LayerNorm(config.out_channels)

        self.fusion_projector = ProjectionHead(
            config.out_channels * 5, config.hidden_channels, config.projection_size, config
        )

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_channels, config.hidden_channels // 2),
            nn.BatchNorm1d(config.hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_channels // 2, config.num_classes)
        )

        self.individual_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.out_channels, config.hidden_channels),
                nn.BatchNorm1d(config.hidden_channels),
                nn.ReLU(),
                nn.Linear(config.hidden_channels, config.num_classes)
            ) for _ in range(5)
        ])

        self.interaction = FeatureInteraction(config.out_channels * 5, config.hidden_channels)

        self.network_performance = [0.5] * 5
        self.performance_history = []
        self.last_update_epoch = -1

    def update_network_weights(self, new_performance, current_epoch):
        if current_epoch - self.last_update_epoch < self.config.weight_update_freq:
            return

        self.last_update_epoch = current_epoch
        alpha = self.config.weight_smoothing
        self.network_performance = [
            alpha * new_p + (1 - alpha) * old_p
            for new_p, old_p in zip(new_performance, self.network_performance)
        ]

        min_perf = min(self.network_performance)
        adjusted_perf = [
            max(p, min_perf * self.config.min_network_weight * 2)
            for p in self.network_performance
        ]

        total = sum(adjusted_perf)
        self.config.network_weights = [p / total for p in adjusted_perf]

        self.performance_history.append(copy.deepcopy(self.config.network_weights))
        print(f"Updated network weights at epoch {current_epoch}: {self.config.network_weights}")

    def evaluate_network_performance(self, networks, labels, mask):
        performance = []
        with torch.no_grad():
            for i in range(5):
                z = self.encoders[i](networks[i].x, networks[i].edge_index)
                logits = self.individual_classifiers[i](z)
                preds = torch.sigmoid(logits[mask]).cpu().numpy()
                trues = labels[mask].cpu().numpy()
                try:
                    roc_auc = metrics.roc_auc_score(trues, preds)
                except ValueError:
                    roc_auc = 0.5
                performance.append(roc_auc)
        return performance

    def forward(self, networks, train_mode=True, hard_sample_mask=None):
        batch_size = networks[0].x.size(0)

        if train_mode:
            augmented_networks = []
            for i, data in enumerate(networks):
                drop_edge_rate = self.config.drop_edge_rates[i]
                drop_feature_rate = self.config.drop_feature_rates[i]
                weight_factor = self.config.network_weights[i] / max(self.config.network_weights)
                drop_edge_rate *= (1 + weight_factor * 0.5)
                drop_feature_rate *= (1 + weight_factor * 0.5)

                if hard_sample_mask is not None and torch.any(hard_sample_mask):
                    x_hard = F.dropout(
                        data.x[hard_sample_mask],
                        p=min(drop_feature_rate * self.config.hard_sample_boost, self.config.max_drop_rate),
                        training=self.training
                    )
                    x_normal = F.dropout(
                        data.x[~hard_sample_mask],
                        p=drop_feature_rate,
                        training=self.training
                    )
                    x = torch.zeros_like(data.x)
                    x[hard_sample_mask] = x_hard
                    x[~hard_sample_mask] = x_normal
                else:
                    x = F.dropout(data.x, p=drop_feature_rate, training=self.training)

                edge_index = dropout_adj(
                    data.edge_index,
                    p=min(drop_edge_rate * (self.config.hard_sample_boost if hard_sample_mask is not None else 1),
                          self.config.max_drop_rate),
                    force_undirected=True
                )[0]
                augmented_networks.append((x, edge_index))
        else:
            augmented_networks = [(data.x, data.edge_index) for data in networks]

        z_list = []
        for i, (x, edge_index) in enumerate(augmented_networks):
            z = self.encoders[i](x, edge_index)
            z_list.append(z)

        enhanced_z_list = []
        for i, z_i in enumerate(z_list):
            z_i_seq = z_i.unsqueeze(1)
            enhanced = z_i_seq
            for j, z_j in enumerate(z_list):
                if i != j:
                    z_j_seq = z_j.unsqueeze(1)
                    attn_output = self.cross_attention(enhanced, z_j_seq, z_j_seq)
                    enhanced = self.layer_norm1(enhanced + attn_output)
            enhanced_z_list.append(enhanced.squeeze(1))

        proj_list = [self.projectors[i](z) for i, z in enumerate(enhanced_z_list)]

        combined_z = torch.cat(enhanced_z_list, dim=1)
        combined_z = self.interaction(combined_z)

        gate_weights, router_z_loss, load_balancing_loss = self.gating(combined_z, self.training)

        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_in = enhanced_z_list[i % len(enhanced_z_list)]
            expert_out = expert(expert_in)
            expert_outputs.append(expert_out)

        stacked_expert_outputs = torch.stack(expert_outputs, dim=1)
        weighted_expert_outputs = gate_weights.unsqueeze(-1) * stacked_expert_outputs
        final_representation = weighted_expert_outputs.sum(dim=1)

        logits = self.classifier(final_representation)
        fusion_proj = self.fusion_projector(combined_z)

        moe_loss = None
        if self.config.use_auxiliary_loss and self.training:
            moe_loss = {
                'router_z_loss': router_z_loss * self.config.router_z_loss_coef,
                'load_balancing_loss': load_balancing_loss * self.config.load_balancing_loss_coef
            }

        return z_list, proj_list, fusion_proj, logits.squeeze(), gate_weights, moe_loss