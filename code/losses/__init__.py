import torch
import torch.nn as nn


class SumOfLosses(nn.Module):
    """Helper class to provide a sum of losses with coefficients

    Args:
        losses (list of torch.nn.Module): list of losses as torch.nn.Module
        coeffs (list of floats): list of coefficients to multiply the losses:
            `total_loss = sum([coeffs[i] * losses[i] for i in N])
        names (list of str): names to output in the dictionary

    """
    def __init__(self, losses, coeffs, names=None, total_loss_name="batchloss"):
        assert isinstance(losses, (list, tuple)) and len(losses) > 0
        assert isinstance(coeffs, (list, tuple)) and len(coeffs) > 0
        assert len(losses) == len(coeffs)
        if names is not None:
            assert isinstance(names, (list, tuple)) and len(names) == len(losses)
        else:
            names = ['loss_{}'.format(i) for i in range(len(losses))]
        super(SumOfLosses, self).__init__()
        self.losses = losses
        self.coeffs = coeffs
        self.names = names
        self.total_loss_name = total_loss_name

    def to(self, *args, **kwargs):
        for l in self.losses:
            l.to(*args, **kwargs)
        return self

    def forward(self, y_pred, y):
        loss_results = [l(y_pred, y) for l in self.losses]
        total_loss = torch.sum(
            torch.cat([c * l.unsqueeze(0) for c, l in zip(self.coeffs, loss_results)], dim=0),
            dim=0
        )
        return {n: v for n, v in zip([self.total_loss_name, *self.names], [total_loss, *loss_results])}
