import torch
from torch.autograd import grad

__all__ = [
    "collect_neighbors",
    "general_derivative"
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
