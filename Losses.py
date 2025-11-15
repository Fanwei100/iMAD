"""Loss functions and helpers tailored to confidence calibration tasks."""

import json
import numpy as np
import torch.nn as nn
import torch
from sklearn.utils.class_weight import compute_class_weight

class FocusCalLoss(nn.Module):
    """Combination of asymmetric focal, calibration, and confidence penalties."""
    def __init__(self, alpha_pos=1.0, alpha_neg=2.0, gamma=2.0, tau=0.7, lambda_cp=0.3, mu_ece=0.1, n_bins=15):
        super().__init__()
        self.alpha_pos = float(alpha_pos)  # α1 (for y=1)
        self.alpha_neg = float(alpha_neg)  # α0 (for y=0)
        self.gamma = gamma          # Focusing parameter γ
        self.tau = tau              # Confidence threshold τ
        self.lambda_cp = lambda_cp  # Weight for confidence penalty
        self.mu_ece = mu_ece        # Weight for ECE
        self.n_bins = n_bins

    def __repr__(self):
        return "FocusCalLoss:-" + json.dumps({
            "alpha_pos": self.alpha_pos,
            "alpha_neg": self.alpha_neg,
            "gamma": self.gamma,
            "tau": self.tau,
            "lambda_cp": self.lambda_cp,
            "mu_ece": self.mu_ece,
            "n_bins": self.n_bins
        })

    def forward(self, logits, targets):
        """Compute the composite loss for predicted correctness and hesitation.

        Args:
            logits (tuple[torch.Tensor, torch.Tensor]): Tuple of ``(p_hat, u_hat)``
                predictions in the range (0, 1).
            targets (torch.Tensor): Binary ground-truth labels with shape ``(N,)``.

        Returns:
            torch.Tensor: Scalar loss averaged across the batch.
        """
        assert isinstance(logits, tuple), "Expected tuple of (p_hat, u_hat)"
        p_hat, u_hat = logits

        p_hat = p_hat.view(-1)
        u_hat = u_hat.view(-1)
        targets = targets.float()

        # -------- Asymmetric Focal Loss  --------
        pt = torch.where(targets == 1, p_hat, 1 - p_hat)
        alpha = torch.where(targets == 1, self.alpha_pos, self.alpha_neg).to(p_hat.device)

        focal_loss = -alpha * ((1 - pt) ** self.gamma) * torch.log(pt + 1e-12)
        focal_loss = focal_loss.mean()

        # -------- Confidence Penalty  --------
        # Penalize underconfident correct and overconfident wrong
        penalty_cp = torch.zeros_like(p_hat)

        mask_overconfident_wrong = (targets == 0) & (p_hat > self.tau)
        mask_underconfident_correct = (targets == 1) & (p_hat < self.tau)

        penalty_cp[mask_overconfident_wrong] = (u_hat[mask_overconfident_wrong]) ** 2
        penalty_cp[mask_underconfident_correct] = (1 - u_hat[mask_underconfident_correct]) ** 2

        cp_loss = penalty_cp.mean()

        # -------- Expected Calibration Error --------
        ece = self._expected_calibration_error(p_hat, targets)

        # -------- Total Loss --------
        total_loss = focal_loss + self.lambda_cp * cp_loss + self.mu_ece * ece

        return total_loss

    def _expected_calibration_error(self, probs, labels):
        """Approximate the Expected Calibration Error (ECE) for binary logits."""
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=probs.device)
        ece = torch.zeros(1, device=probs.device)
        N = probs.size(0)

        for i in range(self.n_bins):
            mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
            if mask.any():
                acc = labels[mask].float().sum()
                conf = probs[mask].sum()
                ece += torch.abs(acc - conf) / N
        return ece


def getClassweights(y_train):
    """Compute class weights aligned with sklearn's ``compute_class_weight``."""
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print("Using ClassWeights",class_weights_tensor)
    return class_weights_tensor

def getLoss(lossName,ClassWeights,y_train,device,SmotType,alpha_pos=1.0, alpha_neg=2.0, gamma=2.0, tau=0.7, lambda_cp=0.3, mu_ece=0.1, n_bins=15):
    """Factory helper returning a configured loss function and metadata."""
    assert  lossName in ("FocusCalLoss",)
    if lossName=="FocusCalLoss":
        if ClassWeights:
            alpha_neg,alpha_pos=getClassweights(y_train)
        loss_fn = FocusCalLoss(alpha_pos=alpha_pos, alpha_neg=alpha_neg, gamma=gamma, tau=tau, lambda_cp=lambda_cp, mu_ece=mu_ece, n_bins=n_bins).to(device)
    return loss_fn,ClassWeights,lossName

