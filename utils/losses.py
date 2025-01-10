import torch
import torchaudio
from librosa.filters import mel as librosa_mel_fn
from auraloss.freq import MultiResolutionSTFTLoss, RandomResolutionSTFTLoss


def get_frequency_weighting(freqs, freq_weighting=None):
    if freq_weighting is None:
        return torch.ones_like(freqs).to(freqs.device)
    elif freq_weighting == "sqrt":
        return torch.sqrt(freqs)
    elif freq_weighting == "exp":
        freqs = torch.exp(freqs)
        return freqs - freqs[:, 0, :].unsqueeze(-2)
    elif freq_weighting == "log":
        return torch.log(1 + freqs)
    elif freq_weighting == "linear":
        return freqs


def get_loss(loss_args, operator=None):
    print("loss_args", loss_args)
    if loss_args.name == "none":
        return None

    if hasattr(loss_args, "loss_1"):  # We have a hybrid of multiple losses
        return lambda x, x_hat: torch.sum(
            torch.stack(
                [
                    get_loss(getattr(loss_args, key), operator=operator)(x, x_hat)
                    for key in list(loss_args.keys())
                ]
            )
        )

    else:
        if "stft" in loss_args.name or "STFT" in loss_args.name:

            def loss_fn(x, x_hat, freq_weighting=None):

                use_STFT_operator = True
                try:
                    use_STFT_operator = loss_args.get("use_STFT_operator", False)
                except:
                    pass

                if use_STFT_operator:
                    assert operator is not None, "Operator must be provided"

                if use_STFT_operator:
                    X = operator.apply_stft(x)
                    X_hat = operator.apply_stft(x_hat)
                    X_hat_flipped = operator.apply_stft(-1 * x_hat)
                    X_hat_sum = operator.apply_stft(x - 1 * x_hat)
                else:
                    win_length = loss_args.get("win_length", 1024)
                    fft_size = loss_args.get("fft_size", 2048)
                    hop_size = loss_args.get("hop_size", 256)
                    window = torch.hann_window(win_length).float().to(x.device)
                    X = torch.stft(
                        x,
                        n_fft=fft_size,
                        hop_length=hop_size,
                        win_length=win_length,
                        window=window,
                        center=True,
                        return_complex=True,
                    )
                    X_hat = torch.stft(
                        x_hat,
                        n_fft=fft_size,
                        hop_length=hop_size,
                        win_length=win_length,
                        window=window,
                        center=True,
                        return_complex=True,
                    )
                    X_hat_flipped = torch.stft(
                        -1 * x_hat,
                        n_fft=fft_size,
                        hop_length=hop_size,
                        win_length=win_length,
                        window=window,
                        center=True,
                        return_complex=True,
                    )
                    X_hat_sum = torch.stft(
                        x + 1 * x_hat,
                        n_fft=fft_size,
                        hop_length=hop_size,
                        win_length=win_length,
                        window=window,
                        center=True,
                        return_complex=True,
                    )

                freqs = (
                    torch.linspace(0, 1, X.shape[-2])
                    .to(X.device)
                    .unsqueeze(-1)
                    .unsqueeze(0)
                    .expand(X.shape)
                    + 1
                )
                freqs = get_frequency_weighting(
                    freqs, freq_weighting=loss_args.get("freq_weighting", None)
                )

                X = X * freqs
                X_hat = X_hat * freqs
                if loss_args.name == "l2_stft_sum":
                    loss = torch.sum((X - X_hat).abs() ** 2)
                elif loss_args.name == "l2_stft_mag_sum":
                    loss = torch.sum((X.abs() - X_hat.abs()) ** 2)
                elif loss_args.name == "l2_abs_stft":
                    loss = torch.linalg.norm(
                        X_hat.abs().reshape(-1) - X.abs().reshape(-1), ord=2
                    )
                    reduction = loss_args.get("reduction", "mean")
                    if reduction == "mean":
                        loss /= X.abs().numel()
                    elif reduction == "sum":
                        pass
                    else:
                        raise NotImplementedError(
                            f"reduction {reduction} not implemented"
                        )

                elif loss_args.name == "l2_stft_mag_summean":
                    loss = torch.mean(torch.sum((X.abs() - X_hat.abs()) ** 2, dim=-2))

                elif loss_args.name == "l2_stft_mag_BABE":
                    loss = torch.linalg.norm(X.abs() - X_hat.abs(), ord=2)

                elif loss_args.name == "l2_stft_mag_summean_sqrt":
                    loss = torch.sqrt(
                        torch.mean(torch.sum((X.abs() - X_hat.abs()) ** 2, dim=-2))
                    )

                elif loss_args.name == "l2_stft_logmag_sum":
                    loss = torch.sum(
                        (torch.log10(X.abs() + 1e-8) - torch.log10(X_hat.abs() + 1e-8))
                        ** 2
                    )

                elif loss_args.name == "l2_comp_stft_sum":
                    compression_factor = loss_args.get("compression_factor", None)
                    assert (
                        compression_factor is not None
                        and compression_factor > 0.0
                        and compression_factor <= 1.0
                    ), f"Compression factor weird: {compression_factor}"
                    # compression_factor = 1
                    X_comp = (X.abs() + 1e-8) ** compression_factor * torch.exp(
                        1j * X.angle()
                    )
                    X_hat_comp = (X_hat.abs() + 1e-8) ** compression_factor * torch.exp(
                        1j * X_hat.angle()
                    )
                    loss = torch.sum((X_comp - X_hat_comp).abs() ** 2)
                elif loss_args.name == "l2_comp_stft_mean":
                    compression_factor = loss_args.get("compression_factor", None)
                    assert (
                        compression_factor is not None
                        and compression_factor > 0.0
                        and compression_factor <= 1.0
                    ), f"Compression factor weird: {compression_factor}"
                    X_comp = (X.abs() + 1e-8) ** compression_factor * torch.exp(
                        1j * X.angle()
                    )
                    X_hat_comp = (X_hat.abs() + 1e-8) ** compression_factor * torch.exp(
                        1j * X_hat.angle()
                    )
                    loss = torch.mean((X_comp - X_hat_comp).abs() ** 2)
                elif "l2_comp_abs_stft_summean" in loss_args.name:
                    compression_factor = loss_args.get("compression_factor", None)
                    assert (
                        compression_factor is not None
                        and compression_factor > 0.0
                        and compression_factor <= 1.0
                    ), f"Compression factor weird: {compression_factor}"
                    X_comp = (X.abs() + 1e-8) ** compression_factor * torch.exp(
                        1j * X.angle() * 0
                    )
                    X_hat_comp = (X_hat.abs() + 1e-8) ** compression_factor * torch.exp(
                        1j * X_hat.angle() * 0
                    )
                    loss = torch.mean(
                        torch.sum((X_comp - X_hat_comp).abs() ** 2, dim=-2)
                    )
                elif "l2_comp_stft_summean" in loss_args.name:
                    compression_factor = loss_args.get("compression_factor", None)
                    assert (
                        compression_factor is not None
                        and compression_factor > 0.0
                        and compression_factor <= 1.0
                    ), f"Compression factor weird: {compression_factor}"
                    X_comp = (X.abs() + 1e-8) ** compression_factor * torch.exp(
                        1j * X.angle()
                    )
                    X_hat_comp = (X_hat.abs() + 1e-8) ** compression_factor * torch.exp(
                        1j * X_hat.angle()
                    )
                    loss = torch.mean(
                        torch.sum((X_comp - X_hat_comp).abs() ** 2, dim=-2)
                    )
                elif "l2_unsquared_comp_stft_summean" in loss_args.name:
                    compression_factor = loss_args.get("compression_factor", None)
                    assert (
                        compression_factor is not None
                        and compression_factor > 0.0
                        and compression_factor <= 1.0
                    ), f"Compression factor weird: {compression_factor}"
                    X_comp = (X.abs() + 1e-8) ** compression_factor * torch.exp(
                        1j * X.angle()
                    )
                    X_hat_comp = (X_hat.abs() + 1e-8) ** compression_factor * torch.exp(
                        1j * X_hat.angle()
                    )
                    loss = torch.sqrt(
                        torch.mean(torch.sum((X_comp - X_hat_comp).abs() ** 2, dim=-2))
                    )

                elif loss_args.name == "l2_log_stft_sum":
                    X_comp = torch.log(1 + X.abs()) * torch.exp(1j * X.angle())
                    X_hat_comp = torch.log(1 + X_hat.abs()) * torch.exp(
                        1j * X_hat.angle()
                    )
                    loss = torch.sum((X_comp - X_hat_comp).abs() ** 2)
                elif loss_args.name == "l2_log_stft_summean":
                    X_comp = torch.log(1 + X.abs()) * torch.exp(1j * X.angle())
                    X_hat_comp = torch.log(1 + X_hat.abs()) * torch.exp(
                        1j * X_hat.angle()
                    )
                    loss = torch.mean(
                        torch.sum((X_comp - X_hat_comp).abs() ** 2, dim=-2)
                    )
                elif loss_args.name == "l2_log_stft_mean":
                    X_comp = torch.log(1 + X.abs()) * torch.exp(1j * X.angle())
                    X_hat_comp = torch.log(1 + X_hat.abs()) * torch.exp(
                        1j * X_hat.angle()
                    )
                    loss = torch.mean((X_comp - X_hat_comp).abs() ** 2)
                elif loss_args.name == "l1_log_abs_stft_summean":
                    X_comp = torch.log(1 + X.abs())
                    X_hat_comp = torch.log(1 + X_hat.abs())
                    loss = torch.mean(torch.sum((X_comp - X_hat_comp).abs(), dim=-2))
                elif loss_args.name == "l1_log10_abs_stft_summean":
                    X_comp = torch.log10(1 + X.abs())
                    X_hat_comp = torch.log10(1 + X_hat.abs())
                    loss = torch.mean(torch.sum((X_comp - X_hat_comp).abs(), dim=-2))
                elif loss_args.name == "l2_mr_stft":
                    loss_stft = MultiResolutionSTFTLoss(
                        fft_sizes=[1024, 2048, 8192],
                        hop_sizes=[256, 512, 2048],
                        win_lengths=[1024, 2048, 8192],
                        sample_rate=44100,
                    )
                    loss = loss_stft(x.unsqueeze(1), x_hat.unsqueeze(1))
                elif loss_args.name == "l2_stft_bin_energy":
                    # Select the most energetic time bin
                    compression_factor = loss_args.get("compression_factor", None)
                    assert (
                        compression_factor is not None
                        and compression_factor > 0.0
                        and compression_factor <= 1.0
                    ), f"Compression factor weird: {compression_factor}"
                    X_comp = (X.abs() + 1e-8) ** compression_factor * torch.exp(
                        1j * X.angle()
                    )
                    X_hat_comp = (X_hat.abs() + 1e-8) ** compression_factor * torch.exp(
                        1j * X_hat.angle()
                    )
                    energy_per_tframe = torch.sum(X_comp.abs(), dim=-2)
                    print(
                        "energy_per_tframe",
                        energy_per_tframe.shape,
                        "Xabs shape:",
                        X.abs().shape,
                    )
                    max_idx = torch.argmax(energy_per_tframe, dim=-1)
                    print("max_idx", max_idx.shape)
                    loss = torch.mean(
                        (X_comp[:, :, max_idx].abs() - X_hat_comp[:, :, max_idx].abs())
                        ** 2
                    )

                elif loss_args.name == "l2_stft_phase":
                    loss = torch.mean(
                        ((X.abs() ** 2 + X_hat.abs() ** 2) - X_hat_sum.abs() ** 2) ** 2
                    )

                elif loss_args.name == "l2_stft_random_resolution":
                    loss_stft = RandomResolutionSTFTLoss(
                        min_fft_size=16, max_fft_size=32768, sample_rate=44100
                    )
                    loss = loss_stft(x.unsqueeze(1), x_hat.unsqueeze(1))

                else:
                    raise NotImplementedError(
                        f"rec_loss {loss_args.name} not implemented"
                    )

                weight = loss_args.get("weight", 1.0)

                return weight * loss

            return lambda x, x_hat: loss_fn(x, x_hat)

        elif "mel_magspec" in loss_args.name:
            if "logmel_magspec" in loss_args.name:

                def loss_fn(x, x_hat):
                    n_mels = loss_args.get("n_mels", 160)
                    n_fft = loss_args.get("n_fft", 2048)
                    mel_basis = librosa_mel_fn(
                        sr=44100, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=None
                    )
                    mel_basis = torch.from_numpy(mel_basis).float().to(x.device)
                    hop_length = n_fft // 4
                    spectro = torchaudio.transforms.Spectrogram(
                        n_fft=n_fft, hop_length=hop_length
                    ).to(x.device)
                    eps = 1e-5
                    X = spectro(x).squeeze()
                    X_hat = spectro(x_hat).squeeze()
                    mel_X = torch.matmul(mel_basis, X)
                    mel_X_hat = torch.matmul(mel_basis, X_hat)
                    mel_X = torch.log10(mel_X + eps)
                    mel_X_hat = torch.log10(mel_X_hat + eps)

                    if "l2" in loss_args.name:
                        error = (mel_X - mel_X_hat) ** 2
                    elif "l1" in loss_args.name:
                        error = torch.abs(mel_X - mel_X_hat)

                    weight = loss_args.get("weight", 1.0)

                    return weight * torch.mean(error)

                return lambda x, x_hat: loss_fn(x, x_hat)
            if "sqrtmel_magspec" in loss_args.name:

                def loss_fn(x, x_hat):
                    X = operator.apply_stft(x)
                    X_hat = operator.apply_stft(x_hat)
                    eps = 1e-8
                    X = torch.sqrt(X.abs() + eps)
                    X_hat = torch.sqrt(X_hat.abs() + eps)

                    if "l2" in loss_args.name:
                        error = (X - X_hat).abs() ** 2
                    elif "l1" in loss_args.name:
                        error = torch.abs(X - X_hat)

                    if "_mean" in loss_args.name:
                        return torch.mean(error)
                    elif "_summean" in loss_args.name:
                        return torch.mean(torch.sum(error, dim=-2))

                return lambda x, x_hat: loss_fn(x, x_hat)

        elif "impulse" in loss_args.name:
            if loss_args.name == "impulse_penalty":

                def loss_fn(x, pos_impulse):
                    loss = -torch.mean(x[..., pos_impulse])
                    weight = loss_args.get("weight", 1.0)
                    return weight * loss

                return lambda x, pos_impulse: loss_fn(x, pos_impulse)
            if loss_args.name == "impulse_gaussian_penalty":

                def loss_fn(x, pos_impulse):
                    # Create a Gaussian centered at the impulse position
                    sigma = loss_args.get("sigma", 1.0)
                    time_steps = torch.arange(x.size(1)).float().to(x.device)
                    gaussian_weights = torch.exp(
                        -0.5 * ((time_steps - pos_impulse) / sigma) ** 2
                    )
                    gaussian_weights = (
                        gaussian_weights / gaussian_weights.sum()
                    )  # Normalize weights
                    # Compute the weighted sum of the output
                    print("weights", gaussian_weights[pos_impulse : pos_impulse + 10])
                    weighted_output = x * gaussian_weights.unsqueeze(0)
                    # Loss is the negative of the weighted sum (maximize response at impulse position)
                    loss = -torch.mean(weighted_output.abs())
                    weight = loss_args.get("weight", 1.0)
                    return weight * loss

                return lambda x, pos_impulse: loss_fn(x, pos_impulse)
            if loss_args.name == "impulse_cross_entropy":

                def loss_fn(x, pos_impulse):
                    print("impulse_cross_entropy")
                    target_positions = torch.zeros(x.size(0)).to(x.device) + pos_impulse
                    target_positions = target_positions.long()
                    x = x.abs().softmax(dim=-1)
                    print(
                        "x",
                        x.shape,
                        "target_positions",
                        target_positions.shape,
                        target_positions,
                        x[:, target_positions[0]],
                    )
                    loss = torch.nn.functional.cross_entropy(x, target_positions)
                    print("loss", loss, x[:, 0])
                    weight = loss_args.get("weight", 1.0)
                    return weight * loss

                return lambda x, pos_impulse: loss_fn(x, pos_impulse)

        else:
            if loss_args.name == "l2_sum":

                def loss_fn(x, x_hat):
                    loss = torch.sum((x - x_hat) ** 2)
                    weight = loss_args.get("weight", 1.0)
                    return weight * loss

            elif loss_args.name == "ESR":

                def loss_fn(x, x_hat):
                    loss = torch.norm(x - x_hat) ** 2 / torch.norm(x) ** 2
                    weight = loss_args.get("weight", 1.0)
                    return weight * loss

            elif loss_args.name == "l2_mean":

                def loss_fn(x, x_hat):
                    loss = torch.mean((x - x_hat) ** 2)
                    weight = loss_args.get("weight", 1.0)
                    return weight * loss

            else:
                raise NotImplementedError(f"rec_loss {loss_args.name} not implemented")

            return lambda x, x_hat: loss_fn(x, x_hat)
