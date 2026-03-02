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


class MaskedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask: torch.Tensor) -> None:
        self.mask.data.copy_(mask)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight * self.mask, self.bias)


class AutoregressiveMADE(nn.Module):
    """Minimal MADE-style autoregressive Bernoulli model."""

    def __init__(self, n_sites: int, hidden_size: int = 64):
        super().__init__()
        if n_sites < 2:
            raise ValueError("AutoregressiveMADE requires n_sites >= 2")
        self.n_sites = int(n_sites)
        self.hidden_size = int(hidden_size)

        self.fc1 = MaskedLinear(self.n_sites, self.hidden_size)
        self.fc2 = MaskedLinear(self.hidden_size, self.n_sites)
        self._build_masks()

    def _build_masks(self) -> None:
        # Variable ordering: 1..n
        in_degrees = torch.arange(1, self.n_sites + 1)
        hidden_degrees = torch.tensor(
            [(i % (self.n_sites - 1)) + 1 for i in range(self.hidden_size)],
            dtype=torch.int64,
        )
        out_degrees = torch.arange(1, self.n_sites + 1)

        mask1 = (hidden_degrees[:, None] >= in_degrees[None, :]).float()
        mask2 = (out_degrees[:, None] > hidden_degrees[None, :]).float()
        self.fc1.set_mask(mask1)
        self.fc2.set_mask(mask2)

    def logits(self, bits: torch.Tensor) -> torch.Tensor:
        x = bits.float()
        h = torch.relu(self.fc1(x))
        logits = self.fc2(h)
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
        for i in range(self.n_sites):
            logits = self.logits(samples)
            probs = torch.sigmoid(logits[:, i]).clamp(min=1e-7, max=1.0 - 1e-7)
            bit = torch.bernoulli(probs)
            samples[:, i] = bit
        return samples.long()
