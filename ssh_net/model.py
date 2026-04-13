import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SymplecticHamiltonianDynamic(nn.Module):

    def __init__(self, d_model, d_k=None, d_v=None, dt=0.1, num_heads=8,
                 attn_drop: float = 0.0, evolution_mode: str = "dynamic",
                 evolution_steps: int = 1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k if d_k else d_model // num_heads
        self.d_v = d_v if d_v else d_model // num_heads
        self.num_heads = num_heads
        self.dt = nn.Parameter(torch.tensor(dt))

        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_p = nn.Linear(d_model, d_model, bias=False)

        self.potential_k = nn.Linear(d_model, self.d_k * num_heads, bias=False)
        self.potential_v = nn.Linear(d_model, self.d_v * num_heads, bias=False)

        self.to_output = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        self.attn_drop = nn.Dropout(float(attn_drop))
        self.evolution_mode = evolution_mode.lower()
        self.evolution_steps = int(evolution_steps)
        if self.evolution_mode not in {"dynamic", "static"}:
            raise ValueError(f"Unsupported evolution_mode={evolution_mode}, expected 'dynamic' or 'static'.")
        if self.evolution_steps < 1:
            raise ValueError(f"evolution_steps must be >= 1, got {self.evolution_steps}")

        self.symplectic_gate = nn.Parameter(torch.ones(1))
        self.v_to_k = nn.Linear(self.d_v, self.d_k, bias=False) if self.d_v != self.d_k else nn.Identity()

    def hamiltonian_evolution(self, q, p, k, v):

        batch_size, seq_len, _ = q.shape

        q_multi = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        p_multi = p.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k_multi = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v_multi = v.view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)

        energy_scores1 = torch.matmul(q_multi, k_multi.transpose(-2, -1)) / math.sqrt(self.d_k)
        energy_weights1 = self.attn_drop(F.softmax(energy_scores1, dim=-1))
        potential_energy1 = torch.matmul(energy_weights1, v_multi)

        if self.evolution_mode == "static":
            q_new_multi = self.v_to_k(potential_energy1)
            p_new_multi = p_multi

            q_new = q_new_multi.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            p_new = p_new_multi.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            return q_new, p_new, potential_energy1

        dV_dq_approx = self.v_to_k(torch.matmul(energy_weights1, v_multi))

        p_half = p_multi - (self.dt / 2) * dV_dq_approx * self.symplectic_gate

        q_new_multi = q_multi + self.dt * p_half

        energy_scores2 = torch.matmul(q_new_multi, k_multi.transpose(-2, -1)) / math.sqrt(self.d_k)
        energy_weights2 = self.attn_drop(F.softmax(energy_scores2, dim=-1))
        potential_energy2 = torch.matmul(energy_weights2, v_multi)

        dV_dq_new_approx = self.v_to_k(torch.matmul(energy_weights2, v_multi))

        p_new_multi = p_half - (self.dt / 2) * dV_dq_new_approx * self.symplectic_gate

        q_new = q_new_multi.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        p_new = p_new_multi.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        avg_potential_energy = (potential_energy1 + potential_energy2) / 2

        return q_new, p_new, avg_potential_energy

    def forward(self, x, return_states=False):

        residual = x

        q = self.to_q(x)
        p = self.to_p(x)

        k = self.potential_k(x)
        v = self.potential_v(x)

        q_new, p_new, potential_energy = q, p, None
        for _ in range(self.evolution_steps):
            q_new, p_new, potential_energy = self.hamiltonian_evolution(q_new, p_new, k, v)

        output = self.to_output(q_new)
        output = self.layer_norm(output + residual)

        if return_states:
            return output, (q_new, p_new, potential_energy)
        return output

class SymplecticStateDescriptor(nn.Module):

    def __init__(self, d_model, d_v, num_heads, output_dim=8):
        super().__init__()
        self.d_model = d_model
        self.d_v = d_v
        self.num_heads = num_heads
        self.output_dim = output_dim

        self.q_proj = nn.Sequential(
            nn.Linear(2 * d_model, 16),
            nn.GELU(),
            nn.Linear(16, output_dim // 3)
        )

        self.p_proj = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, output_dim // 3)
        )

        self.v_proj = nn.Sequential(
            nn.Linear(2 * num_heads * d_v, 16),
            nn.GELU(),
            nn.Linear(16, output_dim - 2 * (output_dim // 3))
        )

        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )

    def forward(self, q, p, v):

        B, L, _ = q.shape

        q_mean = q.mean(dim=1)
        q_var = q.var(dim=1)
        q_desc = torch.cat([q_mean, q_var], dim=1)
        q_desc = self.q_proj(q_desc)

        p_norm = torch.norm(p, dim=-1)
        kinetic_energy = 0.5 * p_norm.mean(dim=1, keepdim=True)
        momentum_dir = p.mean(dim=1)
        momentum_mag = torch.norm(momentum_dir, dim=-1, keepdim=True)
        p_std = p_norm.std(dim=1, keepdim=True)
        momentum_var = momentum_dir.var(dim=-1, keepdim=True)
        p_desc = torch.cat([
            kinetic_energy,
            momentum_mag,
            p_std,
            momentum_var
        ], dim=1)
        p_desc = self.p_proj(p_desc)

        v_merged = v.transpose(1, 2).contiguous().view(B, L, -1)
        v_mean = v_merged.mean(dim=1)
        v_std = v_merged.std(dim=1)
        v_desc = torch.cat([v_mean, v_std], dim=1)
        v_desc = self.v_proj(v_desc)

        combined = torch.cat([q_desc, p_desc, v_desc], dim=1)
        descriptors = self.fusion(combined)

        return descriptors

class SpectralSpatialHamiltonianNet(nn.Module):

    def __init__(self, input_channels=100, num_classes=16, d_model=64,
                 num_hamiltonian_layers=4, num_heads=8, patchsize=15,
                 pos_drop: float = 0.0, attn_drop: float = 0.0,
                 descriptor_dim=8, evolution_mode: str = "dynamic",
                 evolution_steps: int = 1):
        super().__init__()

        self.input_channels = input_channels
        self.d_model = d_model
        self.num_classes = num_classes
        self.patchsize = patchsize
        self.descriptor_dim = descriptor_dim
        self.evolution_mode = evolution_mode
        self.evolution_steps = int(evolution_steps)

        self.spectral_embedding = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, d_model, kernel_size=1, padding=0),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )

        self.spatial_embedding = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )

        self.spatial_position = nn.Parameter(torch.randn(1, patchsize * patchsize, d_model))
        self.pos_drop = nn.Dropout(float(pos_drop))

        self.hamiltonian_layers = nn.ModuleList([
            SymplecticHamiltonianDynamic(
                d_model,
                num_heads=num_heads,
                attn_drop=attn_drop,
                evolution_mode=self.evolution_mode,
                evolution_steps=self.evolution_steps,
            )
            for _ in range(num_hamiltonian_layers)
        ])

        self.multiscale_fusion = nn.ModuleList([
            nn.Conv2d(d_model, d_model // 4, kernel_size=1) for _ in range(4)
        ])

        self.spatial_pyramid = nn.ModuleList([
            nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)
            for kernel_size in [1, 3, 5, 7]
        ])

        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 4)
        )

        self.state_descriptor = SymplecticStateDescriptor(
            d_model=d_model,
            d_v=d_model // num_heads,
            num_heads=num_heads,
            output_dim=descriptor_dim
        )

        total_features = d_model + d_model // 4 + descriptor_dim
        self.dynamic_classifier = nn.Sequential(
            nn.Linear(total_features, d_model),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, num_classes)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):

        if x.dim() == 5:
            x = x.squeeze()
        batch_size = x.shape[0]

        spectral_features = self.spectral_embedding(x)
        spatial_features = self.spatial_embedding(spectral_features)
        spatial_seq = spatial_features.view(batch_size, self.d_model, -1).transpose(1, 2)
        spatial_seq = spatial_seq + self.spatial_position
        spatial_seq = self.pos_drop(spatial_seq)

        hamiltonian_states = []
        current_seq = spatial_seq
        for hamiltonian_layer in self.hamiltonian_layers:
            current_seq, states = hamiltonian_layer(current_seq, return_states=True)
            hamiltonian_states.append(states)

        dynamic_features = current_seq.transpose(1, 2).view(
            batch_size, self.d_model, self.patchsize, self.patchsize
        )

        pyramid_features = []
        for conv, pool in zip(self.multiscale_fusion, self.spatial_pyramid):
            pooled = pool(dynamic_features)
            fused = conv(pooled)
            pyramid_features.append(fused)

        multiscale_feat = torch.cat([
            F.adaptive_avg_pool2d(feat, (1, 1)).view(batch_size, -1)
            for feat in pyramid_features
        ], dim=1)

        global_feat = self.global_context(dynamic_features)

        q_final, p_final, v_final = hamiltonian_states[-1]

        dynamic_descriptors = self.state_descriptor(q_final, p_final, v_final)

        combined_features = torch.cat([
            multiscale_feat,
            global_feat,
            dynamic_descriptors
        ], dim=1)

        output = self.dynamic_classifier(combined_features)
        return output

__all__ = [
    "SymplecticHamiltonianDynamic",
    "SymplecticStateDescriptor",
    "SpectralSpatialHamiltonianNet",
]
