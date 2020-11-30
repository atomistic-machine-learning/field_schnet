import torch
import torch.nn as nn
from torch.autograd import grad
import schnetpack as spk

__all__ = [
    "collect_neighbors",
    "general_derivative",
    "AtomDistances"
]


def collect_neighbors(x, neighbors):
    """
    Get properties of neighbors
    """
    nbh_size = neighbors.size()
    nbh = neighbors.view(-1, nbh_size[1] * nbh_size[2], 1)
    nbh = nbh.expand(-1, -1, x.size(2))

    y = torch.gather(x, 1, nbh)
    y = y.view(nbh_size[0], nbh_size[1], nbh_size[2], -1)

    return y


def general_derivative(fx, dx, hessian=False, create_graph=True, retain_graph=True):
    fx_shape = fx.size()
    dx_shape = dx.size()

    # print('=======================')
    # print(fx_shape, 'FX')
    # print(dx_shape, 'DX')

    fx = fx.view(fx_shape[0], -1)
    # print(fx.shape, 'REFORM')

    dfxdx = torch.stack(
        [grad(fx[..., i], dx, torch.ones_like(fx[:, i]), create_graph=create_graph, retain_graph=retain_graph)[0] for
         i in range(fx.size(1))], dim=1
    )

    if not hessian:
        dfxdx = dfxdx.view(*fx_shape, *dx_shape[1:])
        # TODO: squeeze for energies might be dangerous
        dfxdx = dfxdx.squeeze(dim=1)  # Saver squeeze for batch size of 1
    else:
        dfxdx = dfxdx.view(fx_shape[0], fx_shape[-1] * fx_shape[-2], -1)

    # print(dfxdx.shape, 'DFXDX')
    # print('=======================')

    return dfxdx


class AtomDistances(nn.Module):
    r"""Layer for computing distance of every atom to its neighbors.

    Args:
        return_directions (bool, optional): if True, the `forward` method also returns
            normalized direction vectors.

    """

    def __init__(self, return_directions=False, normalize_vecs=True):
        super(AtomDistances, self).__init__()
        self.return_directions = return_directions
        self.normalize_vecs = normalize_vecs

    def forward(
        self, positions, neighbors, cell=None, cell_offsets=None, neighbor_mask=None
    ):
        r"""Compute distance of every atom to its neighbors.

        Args:
            positions (torch.Tensor): atomic Cartesian coordinates with
                (N_b x N_at x 3) shape.
            neighbors (torch.Tensor): indices of neighboring atoms to consider
                with (N_b x N_at x N_nbh) shape.
            cell (torch.tensor, optional): periodic cell of (N_b x 3 x 3) shape.
            cell_offsets (torch.Tensor, optional): offset of atom in cell coordinates
                with (N_b x N_at x N_nbh x 3) shape.
            neighbor_mask (torch.Tensor, optional): boolean mask for neighbor
                positions. Required for the stable computation of forces in
                molecules with different sizes.

        Returns:
            torch.Tensor: layer output of (N_b x N_at x N_nbh) shape.

        """
        return spk.nn.atom_distances(
            positions,
            neighbors,
            cell,
            cell_offsets,
            return_vecs=self.return_directions,
            normalize_vecs=self.normalize_vecs,
            neighbor_mask=neighbor_mask,
        )