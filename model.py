import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class AttentionExpertGate(nn.Module):
    def __init__(self, dim, num_experts=4, scale=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.scale = scale

        # -------- expert residual projections --------
        self.experts = nn.ModuleList([
            nn.Linear(dim, dim, bias=False)
            for _ in range(num_experts)
        ])

        # -------- attention over experts --------
        self.attn = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.Linear(dim, num_experts)
        )

    def forward(self, gru_feat, prior_feat):
        # base fusion (same as without_Adaptive_fusion)
        base = 0.5 * gru_feat + 0.5 * prior_feat

        # difference signal
        diff = gru_feat - prior_feat

        # expert residuals
        expert_residuals = torch.stack(
            [expert(diff) for expert in self.experts],
            dim=1
        )  # [B, K, D]

        # attention weights
        fused = torch.cat([gru_feat, prior_feat], dim=-1)
        weights = F.softmax(self.attn(fused), dim=-1)  # [B, K]

        # weighted residual
        delta = torch.sum(
            weights.unsqueeze(-1) * expert_residuals,
            dim=1
        )

        return base + self.scale * delta



# ==================================================
# 主模型
# ==================================================
class LLMSAN(nn.Module):
    def __init__(self, seq_input_dim, mean_input_dim, prior_input_dim,
                 hidden_dim, output_dim, ablation_type="LLMSAN"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.ablation_type = ablation_type

        # --- GRU model ---
        self.gru = nn.GRU(seq_input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc_mean = nn.Linear(mean_input_dim, hidden_dim)
        self.gru_fc = nn.Linear(hidden_dim * 2, output_dim)

        # --- Prior model 模块 ---
        self.prior_gcn1 = GCNConv(prior_input_dim, hidden_dim)
        self.prior_gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.prior_fc = nn.Linear(hidden_dim, output_dim)

        self.fusion_gate = AttentionExpertGate(output_dim)


    def forward(self, input_seq, mean_embedding, prior_target):
        B = input_seq.size(0)
        device = input_seq.device

        # =================
        # GRU embedding
        # =================
        gru_out, _ = self.gru(input_seq)
        last_hidden = gru_out[:, -1, :]
        mean_feat = F.relu(self.fc_mean(mean_embedding))
        gru_feat = torch.cat([last_hidden, mean_feat], dim=1)
        gru_out_proj = self.gru_fc(gru_feat)


        # =================
        # GCN embedding
        # =================
 
        sim_matrix = torch.matmul(prior_target, prior_target.T)
        sim_matrix.fill_diagonal_(0)
        num_edges = max(1, int(B * B * 0.05))
        flat_sim = sim_matrix.flatten()
        topk_values, topk_indices = torch.topk(flat_sim, num_edges)
        row = topk_indices // B
        col = topk_indices % B
        edge_index = torch.stack([row, col], dim=0).long()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        prior_hidden = F.relu(self.prior_gcn1(prior_target, edge_index))
        prior_hidden = F.relu(self.prior_gcn2(prior_hidden, edge_index))
        prior_out = self.prior_fc(prior_hidden)


        # =================
        # fusion
        # =================

        pred = self.fusion_gate(gru_out_proj, prior_out)


        return pred, gru_out_proj, prior_out
