import torch


def extend_grid(grid):
    N = grid.shape[0]
    n = 2 / N
    x = torch.concatenate((torch.tensor([-1 - n]), grid.cpu(), torch.tensor([1 + n])))
    pts = x.clone()
    return pts


def calculate_mu_grid(mu, G, device):
    grid = torch.linspace(-1, 1, steps=G, device=device, dtype=torch.float32)
    grid = torch.sign(grid) * (((1 + mu) ** (grid.abs()) - 1) / mu)
    grid = extend_grid(grid)[None, :]
    return grid


def CCR_Basis(t, alpha):
    """ "
    Quartic catmull rom basis
    """
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t

    f_0 = 0.5 * (-t + 2 * (1 + alpha) * t2 - (1 + 4 * alpha) * t3 + 2 * alpha * t4)
    f_1 = 0.5 * (2 - (5 + 2 * alpha) * t2 + (3 + 4 * alpha) * t3 - 2 * alpha * t4)
    f_2 = 0.5 * (t + 2 * (2 - alpha) * t2 - (3 - 4 * alpha) * t3 - 2 * alpha * t4)
    f_3 = 0.5 * (-1 * (1 - 2 * alpha) * t2 + (1 - 4 * alpha) * t3 + 2 * alpha * t4)
    f = torch.stack((f_0, f_1, f_2, f_3), dim=1)
    return f


def BSpline_Basis(t):
    """
    B-Spline basis
    """
    t2 = t * t
    t3 = t2 * t
    f_0 = 1 / 6 * (1 - 3 * t + 3 * t2 - t3)
    f_1 = 1 / 6 * (4 - 6 * t2 + 3 * t3)
    f_2 = 1 / 6 * (1 + 3 * t + 3 * t2 - 3 * t3)
    f_3 = 1 / 6 * t3
    f = torch.stack((f_0, f_1, f_2, f_3), dim=1)
    return f


def calculate_CCR(x_eval, grid, alphas, coefs, device="cpu"):
    """
    Calculate the CCR curve
    :param x_eval: torch.tensor, shape (N, 1)
    :param grid: torch.tensor, shape (N, 1)
    :param alphas: torch.tensor, shape (N,)
    :param coefs: torch.tensor, shape (N,)
    """
    # If x_eval is in a grid
    x_in_grid = (x_eval >= grid[:, :-1]) * (x_eval < grid[:, 1:]).to(device)
    # x_in_grid[-1, -1] = True
    interval_differences = torch.diff(grid).to(device)
    t_for_curve_segments = x_eval * x_in_grid.to(torch.int).to(
        device
    )  # Change boolean to sample values on positions
    # recalculate t for segment to be between 0-1
    t_for_curve_segments = (t_for_curve_segments - grid[:, :-1]) / interval_differences
    t_for_curve_segments = t_for_curve_segments * x_in_grid  # this is maybe not needed

    y = torch.zeros_like(x_eval).to(device)
    basis_b = torch.zeros((x_eval.shape[0], 4)).to(device)
    pts_temp = torch.zeros((x_eval.shape[0], 4)).to(device)
    for col in range(1, coefs.shape[0] - 2):
        t = t_for_curve_segments[(x_in_grid[:, col] == True), col]
        t_idx = torch.nonzero(x_in_grid[:, col]).squeeze()
        if t.shape[-1] != 0:
            # calculate mapping
            basis = CCR_Basis(t, alphas[col - 1])
            basis_b[t_idx] = basis.clone()
            pts_temp[t_idx] = coefs[col - 1 : col + 3].clone()

    y = torch.sum(basis_b * pts_temp, dim=1).requires_grad_(True)
    return y


def _test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    coefs = torch.rand(1, 11, device=device, dtype=torch.float32)
    grid = torch.linspace(-1, 1, steps=11)
    grid = extend_grid(grid)[:, None]
    grid = grid[1:-2, 0][None, :].to(device)
    alphas = torch.rand(1, 11, device=device, dtype=torch.float32)
    params_nld = {"spline_coefs": torch.nn.Parameter(coefs, requires_grad=True)}  # ,
    x = torch.linspace(-1, 1, 2**18)[None, :, None].to(device)
    y = x.clone()
    print(f"x shape {x.shape}")
    for i in range(x.shape[0]):
        print(x[i].shape)
        y[i, :] = calculate_CCR(
            x_eval=x[i][:, None],
            grid=grid,
            alphas=alphas,
            coefs=params_nld["spline_coefs"],
            device=device,
        ).unsqueeze(1)
    print(y.shape)


if __name__ == "__main__":
    _test()
