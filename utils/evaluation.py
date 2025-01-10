from librosa.filters import mel as librosa_mel_fn
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchaudio


class val_loss_supervised(nn.Module):
    def __init__(
        self,
        device="cpu",
        spec_scales=[2048, 1024, 512, 256, 128, 64],
        mismatched_DI=False,
        clip_len=12,
    ):
        super().__init__()
        self.spec_scales = spec_scales
        self.specs = [
            TimeFreqConverter(
                n_fft=scale,
                hop_length=scale // 4,
                win_length=scale,
                sampling_rate=44100,
                n_mel_channels=160,
            ).to(device)
            for scale in self.spec_scales
        ]
        self.mel_spec = TimeFreqConverter(
            n_fft=2048,
            hop_length=2048 // 4,
            win_length=2048,
            sampling_rate=44100,
            n_mel_channels=160,
        ).to(device)
        self.log_eps = 1e-5
        self.losses = {
            "ms_spec_loss": 0,
            "ms_log_spec_loss": 0,
            "mel_spec_loss": 0,
            "log_mel_spec_loss": 0,
        }
        self.bst_losses = {
            "ms_spec_loss": [0, 1e9],
            "ms_log_spec_loss": [0, 1e9],
            "mel_spec_loss": [0, 1e9],
            "log_mel_spec_loss": [0, 1e9],
        }
        self.iter_count = 0

        self.mismatched_DI = mismatched_DI
        if self.mismatched_DI:
            self.mel_maker = MelMaker().to(device)
            self.losses = {
                "ms_spec_loss": 0,
                "ms_log_spec_loss": 0,
                "mel_spec_loss": 0,
                "log_mel_spec_loss": 0,
                "long_term_spec_loss": 0,
                "log_long_term_spec_loss": 0,
                "long_term_mel_loss": 0,
                "log_long_term_mel_loss": 0,
            }
            self.bst_losses = {
                "ms_spec_loss": [0, 1e9],
                "ms_log_spec_loss": [0, 1e9],
                "mel_spec_loss": [0, 1e9],
                "log_mel_spec_loss": [0, 1e9],
                "long_term_spec_loss": [0, 1e9],
                "log_long_term_spec_loss": [0, 1e9],
                "long_term_mel_loss": [0, 1e9],
                "log_long_term_mel_loss": [0, 1e9],
            }

    def forward(self, output, target):

        if not self.mismatched_DI:
            losses = {
                "ms_spec_loss": 0,
                "ms_log_spec_loss": 0,
                "mel_spec_loss": 0,
                "log_mel_spec_loss": 0,
            }
        else:
            losses = {
                "ms_spec_loss": 0,
                "ms_log_spec_loss": 0,
                "mel_spec_loss": 0,
                "log_mel_spec_loss": 0,
                "long_term_spec_loss": 0,
                "log_long_term_spec_loss": 0,
                "long_term_mel_loss": 0,
                "log_long_term_mel_loss": 0,
            }

            lmagx = torch.abs(torch.fft.rfft(output))
            lmagy = torch.abs(torch.fft.rfft(target))
            losses["long_term_spec_loss"] = F.l1_loss(lmagx, lmagy)

            loglmagx = torch.log10(torch.clamp(lmagx, self.log_eps))
            loglmagy = torch.log10(torch.clamp(lmagy, self.log_eps))
            losses["log_long_term_spec_loss"] = F.l1_loss(loglmagx, loglmagy)

            lmelx = self.mel_maker(lmagx.unsqueeze(2))
            lmely = self.mel_maker(lmagy.unsqueeze(2))
            losses["long_term_mel_loss"] = F.l1_loss(lmagx, lmagy)

            loglmelx = torch.log10(torch.clamp(lmelx, self.log_eps))
            loglmely = torch.log10(torch.clamp(lmely, self.log_eps))
            losses["log_long_term_mel_loss"] = F.l1_loss(loglmelx, loglmely)

        for spec in self.specs:
            magx = spec(output, mel=False)
            magy = spec(target, mel=False)
            losses["ms_spec_loss"] += F.l1_loss(magx, magy)

            logx = torch.log10(torch.clamp(magx, self.log_eps))
            logy = torch.log10(torch.clamp(magy, self.log_eps))
            losses["ms_log_spec_loss"] += F.l1_loss(logx, logy)

        _, melx = self.mel_spec(output, mel=True)
        _, mely = self.mel_spec(target, mel=True)
        losses["mel_spec_loss"] = F.l1_loss(melx, mely)

        logmelx = torch.log10(torch.clamp(melx, min=self.log_eps))
        logmely = torch.log10(torch.clamp(mely, min=self.log_eps))
        losses["log_mel_spec_loss"] = F.l1_loss(logmelx, logmely)

        return losses

    def add_loss(self, output, target):
        if self.iter_count == 0:
            if not self.mismatched_DI:
                self.losses = {
                    "ms_spec_loss": 0,
                    "ms_log_spec_loss": 0,
                    "mel_spec_loss": 0,
                    "log_mel_spec_loss": 0,
                }
            else:
                self.losses = {
                    "ms_spec_loss": 0,
                    "ms_log_spec_loss": 0,
                    "mel_spec_loss": 0,
                    "log_mel_spec_loss": 0,
                    "long_term_spec_loss": 0,
                    "log_long_term_spec_loss": 0,
                    "long_term_mel_loss": 0,
                    "log_long_term_mel_loss": 0,
                }
        losses = self(output, target)
        for key in losses:
            self.losses[key] += losses[key]
        self.iter_count += 1

    def end_val(self, cur_step):
        for key in self.losses:
            loss = self.losses[key] / self.iter_count
            if loss < self.bst_losses[key][1]:
                self.bst_losses[key] = [cur_step, loss]
        self.iter_count = 0


def multi_mag_loss(
    output, target, fft_sizes=(2048, 1024, 512, 256, 128, 64), log=False
):
    losses = {}
    total_loss = 0
    for fft in fft_sizes:
        hop = fft // 4
        losses[fft] = mag_spec_loss(output, target, fft, hop, log).item()
        total_loss += losses[fft]
    losses["total"] = total_loss
    return losses


def mag_spec_loss(output, target, fft_size=512, hop_size=128, log=False):

    magx = torch.abs(
        torch.stft(output, n_fft=fft_size, hop_length=hop_size, return_complex=True)
    )
    magy = torch.abs(
        torch.stft(target, n_fft=fft_size, hop_length=hop_size, return_complex=True)
    )

    return F.l1_loss(magx, magy)  # , F.l1_loss(logx, logy)


def esr_loss(output, target):
    loss = torch.add(target, -output)
    loss = torch.pow(loss, 2)
    loss = torch.mean(loss)
    energy = torch.mean(torch.pow(target, 2)) + 0.00001
    return torch.div(loss, energy).item()


class TimeFreqConverter(nn.Module):
    def __init__(
        self,
        n_fft=2048,
        hop_length=512,
        win_length=2048,
        sampling_rate=44100,
        n_mel_channels=160,
        mel_fmin=0.0,
        mel_fmax=None,
    ):
        super().__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################
        window = torch.hann_window(win_length).float()
        mel_basis = librosa_mel_fn(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.spectro = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, hop_length=n_fft // 4
        )

    def forward(self, audio, mel=False):
        magnitude = self.spectro(audio).squeeze()
        """p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)

        #magnitude = fft.abs()"""
        if mel:
            mel_output = torch.matmul(self.mel_basis, magnitude)
            return magnitude, mel_output
        else:
            return magnitude
        # if self.log:
        # log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        # return log_mel_spec
        # else:
        # return mel_output


class MelMaker(nn.Module):
    def __init__(
        self,
        clip_len=12,
        sampling_rate=44100,
        n_mel_channels=160,
        mel_fmin=0.0,
        mel_fmax=None,
    ):
        super().__init__()

        mel_basis = librosa_mel_fn(
            sr=sampling_rate,
            n_fft=clip_len * sampling_rate,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, magnitude):
        mel_output = torch.matmul(self.mel_basis, magnitude)
        return mel_output
