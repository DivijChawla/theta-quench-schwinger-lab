from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoregressiveGRU(nn.Module):
    def __init__(self, n_sites: int, hidden_size: int = 64):
        super().__init__()
        self.n_sites = n_sites
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def logits(self, bits: torch.Tensor) -> torch.Tensor:
        bits = bits.float()
        batch = bits.shape[0]
        prev = torch.zeros(batch, self.n_sites, 1, device=bits.device, dtype=bits.dtype)
        prev[:, 1:, 0] = bits[:, :-1]
        h, _ = self.gru(prev)
        logits = self.head(h).squeeze(-1)
        return logits

    def nll(self, bits: torch.Tensor) -> torch.Tensor:
        logits = self.logits(bits)
        bce = F.binary_cross_entropy_with_logits(logits, bits.float(), reduction="none")
        return bce.sum(dim=1)

    def log_prob(self, bits: torch.Tensor) -> torch.Tensor:
        return -self.nll(bits)

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device | str = "cpu") -> torch.Tensor:
        device = torch.device(device)
        samples = torch.zeros(num_samples, self.n_sites, device=device)
        h = None
        prev = torch.zeros(num_samples, 1, 1, device=device)

        for i in range(self.n_sites):
            out, h = self.gru(prev, h)
            logits = self.head(out[:, -1, :]).squeeze(-1)
            probs = torch.sigmoid(logits)
            bit = torch.bernoulli(probs)
            samples[:, i] = bit
            prev = bit.view(num_samples, 1, 1)

        return samples.long()
