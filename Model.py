"""Neural network architectures used by the calibration-aware classifier."""

import torch
import torch.nn as nn

class MLP2Head(nn.Module):
    """Feed-forward encoder with dual heads for correctness and hesitation."""
    def __init__(self, input_dim, nlayers, hidden_dim,output_dim=1, dropout_rate=0.3, use_batchnorm=False, confidencecolumnIndex=-1):
        super(MLP2Head, self).__init__()

        self.confcolidx = confidencecolumnIndex
        # Shared MLP encoder
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        for _ in range(nlayers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        self.shared_encoder = nn.Sequential(*layers)

        # Two heads: correctness & hesitation
        self.correctness_head = nn.Linear(hidden_dim, output_dim)  # ℓ_p
        self.hesitation_head = nn.Linear(hidden_dim, output_dim)   # ℓ_u

        # Learnable combination weights (w1, w2, epsilon)
        self.w1 = nn.Parameter(torch.randn(1))
        self.w2 = nn.Parameter(torch.randn(1))
        self.epsilon = 1e-8

    def Name(self):
        """Return the canonical model identifier."""
        return "MLP2Head"

    def forward(self, x):
        """Run the forward pass and emit correctness and hesitation predictions.

        Args:
            x (torch.Tensor): Features including the LLM confidence column.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of probabilities for
            correctness and hesitation, each with shape ``[batch, 1]``.
        """
        # Extract LLM confidence and compute ℓ_LLM = log(p / (1-p))
        confidense_raw = x[:, self.confcolidx:self.confcolidx+1]  # shape: [batch, 1]
        # Min-max normalize to [0,1]
        confidense = (confidense_raw - confidense_raw.min()) / (confidense_raw.max() - confidense_raw.min() + 1e-8)

        # Optional: Clamp to avoid exact 0 or 1 (numerical safety)
        confidense = torch.clamp(confidense, min=1e-5, max=1 - 1e-5)

        llm_logit = torch.log(confidense / (1 - confidense + 1e-8))

        # Run shared encoder
        shared_repr = self.shared_encoder(x)

        # Compute scalar logits from heads
        ℓ_p = self.correctness_head(shared_repr)  # shape: [batch, 1]
        ℓ_u = self.hesitation_head(shared_repr)   # shape: [batch, 1]

        # Predicted correctness: p̂ = σ(w1 * ℓ_LLM + w2 * ℓ_p + epsilon)
        combined_logit = self.w1 * llm_logit + self.w2 * ℓ_p + self.epsilon
        p_hat = torch.sigmoid(combined_logit)

        # Predicted hesitation/uncertainty: û = σ(ℓ_u)
        u_hat = torch.sigmoid(ℓ_u)

        return p_hat, u_hat