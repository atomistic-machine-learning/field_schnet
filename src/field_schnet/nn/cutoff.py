import torch
from torch import nn


class MollifierCutoff(nn.Module):
    r"""Class for mollifier cutoff scaled to have a value of 1 at :math:`r=0`.

    .. math::
       f(r) = \begin{cases}
        \exp\left(1 - \frac{1}{1 - \left(\frac{r}{r_\text{cutoff}}\right)^2}\right)
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff (float, optional): Cutoff radius.
    """

    def __init__(self, cutoff=5.0):
        super(MollifierCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def forward(self, distances):
        """
        Compute cutoff.

        Args:
            distances (torch.Tensor): values of interatomic distances.

        Returns:
            torch.Tensor: values of cutoff function.
        """
        zeros = torch.zeros_like(distances)
        # Mask distances
        masked_distances = torch.where(distances < self.cutoff, distances, zeros)
        # Compute exponent, alternative forms are:
        #  x^2 / (x^2 - rc^2)
        # -x^2 / [(x-rc)*(x+rc)]
        exponent = 1.0 - 1.0 / (1.0 - torch.pow(masked_distances / self.cutoff, 2))

        return torch.where(distances < self.cutoff, torch.exp(exponent), zeros)
