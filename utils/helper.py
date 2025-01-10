import torch


def find_closest(grid, x_eval):
    """
    Find the closest grid point to the evaluation point x_eval
    grid: grid points, shape (N, )
    x_eval: evaluation points (M,)

    return: assigned grid points, shape (N,1)
    """

    grid = grid[None, :]
    x_eval = x_eval[:, None]  # just to be sure

    # Calculate the distance between grid points and evaluation points
    dist = torch.abs(grid - x_eval)
    min_idx = torch.argmin(dist, dim=0)
    y_eval = x_eval[min_idx].squeeze()
    return y_eval
