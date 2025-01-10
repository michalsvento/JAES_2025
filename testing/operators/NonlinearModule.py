import numpy as np
import torch.nn as nn
import torch
import torchcde
import numpy as np
from scipy.optimize import fsolve
from testing.operators.shared import Operator


class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, num_layers):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features, hidden_features))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_features, hidden_features))
        self.layers.append(nn.Linear(hidden_features, out_features))
        self.act = torch.nn.ReLU()
        # Initialize the parameters of the layers
        self._initialize_weights()

    def _initialize_weights(self):
        # Loop through each layer in the ModuleList
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    #
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        x = self.layers[-1](x)
        return x


class Nonlinear(Operator):

    def __init__(self, op_hp, sample_rate, sample_signal=None, is_dummy=False):
        super().__init__()
        if not (is_dummy):

            self.op_hp = op_hp
            self.sample_rate = sample_rate
            self.op_hp = op_hp
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.sample_signal = sample_signal
            self.n_fft = op_hp.NFFT
            self.win_length = op_hp.win_length
            self.hop_length = op_hp.hop_length
            w = op_hp.window
            if w == "hann":
                self.window = torch.hann_window(self.win_length, device=self.device)
                self.window_h = torch.cat(
                    [
                        self.window,
                        torch.zeros(self.n_fft - self.win_length, device=self.device),
                    ]
                )
                self.window_h = torch.roll(self.window_h, -self.win_length // 2, 0)
            else:
                raise NotImplementedError("window type {} not implemented".format(w))

            if op_hp.NLD == "tanh":
                self.g_pre = nn.Parameter(
                    torch.tensor(op_hp.g_pre, device=self.device, dtype=torch.float32),
                    requires_grad=True,
                )
                self.g_post = nn.Parameter(
                    torch.tensor(op_hp.g_post, device=self.device, dtype=torch.float32),
                    requires_grad=True,
                )
                self.params_nld = {"g_pre": self.g_pre, "g_post": self.g_post}
                self.nld = lambda x: torch.tanh(self.g_pre * x) * self.g_post
            if op_hp.NLD == "softclip":
                g_pre_tensor = (
                    torch.tensor(op_hp.tanh.g_pre).to(self.device).to(torch.float32)
                )
                self.params_nld = {"g_pre": g_pre_tensor}

                def nld(x):
                    g_pre = self.params["g_pre"]
                    g_post = 1.0 / g_pre
                    return torch.tanh(g_pre * x.to(g_pre.device)) * g_post

                self.nld = nld
            elif op_hp.NLD == "hardclip_symmetrical":
                self.limit = nn.Parameter(
                    torch.tensor(op_hp.limit1, device=self.device, dtype=torch.float32),
                    requires_grad=True,
                ).to(self.device)
                self.params_nld = {"limit": self.limit}
                self.nld = lambda x: torch.clip(
                    x,
                    -self.params["limit"].to(x.device),
                    self.params["limit"].to(x.device),
                )

            elif op_hp.NLD == "hardclip":
                g_pre = op_hp.tanh.g_pre
                g_post = op_hp.tanh.g_post
                self.nld = lambda x: torch.clip(x, -op_hp.limit1, op_hp.limit2)
                self.params_nld = {}
            elif op_hp.NLD == "MLP":
                self.model = MLP(
                    in_features=1, out_features=1, hidden_features=20, num_layers=3
                ).to(self.device)

                def nld(x):
                    x_shape = x.shape
                    x = x.view(-1, 1, 1).to(self.device)
                    self.model.load_state_dict(self.params)
                    self.model.to(self.device)
                    x = self.model(x)
                    return x.view(*x_shape)

                self.nld = nld
                self.params_nld = {
                    name: param for name, param in self.model.named_parameters()
                }
            elif op_hp.NLD == "CCR":
                from utils.nonlinear.cubic_catmull_rom import (
                    extend_grid,
                    CCR_Basis,
                    calculate_CCR,
                    calculate_mu_grid,
                )

                self.mu = torch.tensor(op_hp.mu, dtype=torch.float32)
                self.G = op_hp.G
                grid = calculate_mu_grid(self.mu, self.G, device=self.device)
                self.coefs = torch.zeros_like(
                    grid, device=self.device, dtype=torch.float32
                ).view(-1)
                self.grid = grid.to(self.device)
                self.alphas = torch.zeros(
                    op_hp.G - 1, device=self.device, dtype=torch.float32
                )
                self.params_nld = {
                    "spline_coefs": torch.nn.Parameter(self.coefs, requires_grad=True),
                    "alphas": torch.nn.Parameter(self.alphas, requires_grad=True),
                }

                def nld(x):
                    y = x.clone()
                    for i in range(x.shape[0]):
                        y[i, :] = calculate_CCR(
                            x_eval=x[i][:, None],
                            grid=self.grid.to(x.device),
                            alphas=self.alphas.to(x.device),
                            coefs=self.params["spline_coefs"].to(x.device),
                            device=x.device,
                        )
                    return y

                self.nld = nld
            elif op_hp.NLD == "sumtanh":
                order = op_hp.sumtanh.order
                self.a = torch.ones(order + 1, device=self.device, dtype=torch.float32)
                for i in range(order + 1):
                    self.a[i] *= 0.1 * 1 / (i + 1)

                self.params_nld = {"a": torch.nn.Parameter(self.a, requires_grad=True)}

                def nld(x):
                    x = x.to(self.params["a"][0].device)
                    for i in range(order + 1):
                        if i == 0:
                            y = self.params["a"][i] * torch.tanh(x)
                        else:
                            y += self.params["a"][i] * torch.tanh(i * x)
                    return y

                self.nld = nld

            # Nonlearnable NLDs - just for testing
            elif op_hp.NLD == "HWR":
                self.params_nld = {}
                self.nld = lambda x: torch.relu(x)
            elif op_hp.NLD == "FWR":
                self.params_nld = {}
                self.nld = lambda x: torch.abs(x)
            elif op_hp.NLD == "foldback":
                self.limit = nn.Parameter(
                    torch.tensor(op_hp.limit1, device=self.device, dtype=torch.float32),
                    requires_grad=True,
                ).to(self.device)
                self.params_nld = {"limit": self.limit}

                def nld(x):
                    y = x.clone().to(x.device)
                    lim = self.limit.to(x.device)
                    y = torch.where(
                        y.abs() >= lim, torch.sign(y) * (2 * lim - y.abs()), y
                    )
                    return y

                self.nld = nld

            elif op_hp.NLD == "quantize":
                self.w = nn.Parameter(
                    torch.tensor(
                        op_hp.quantize.w, device=self.device, dtype=torch.float32
                    ),
                    requires_grad=True,
                ).to(self.device)
                self.params_nld = {"w": self.w}

                def nld(x, eps=1e-8):
                    # adapted from the original code Pavel Záviška, Brno University of Technology, 2020
                    # to make it compatible with PyTorch
                    y = x.clone().to(x.device)
                    print("self.w", self.w)
                    delta = 2 ** (-self.w + 1)
                    # Mid-tread quantization
                    xq = torch.sign(y) * delta * torch.floor(torch.abs(y) / delta + 0.5)
                    xq[xq > 1] = 1
                    xq[xq < -1] = -1
                    return xq

                self.nld = nld

            elif op_hp.NLD == "carbon_mic":
                self.alpha = nn.Parameter(
                    torch.tensor(op_hp.alpha, device=self.device, dtype=torch.float32)
                )
                g_pre_tensor = (
                    torch.tensor(op_hp.g_pre).to(self.device).to(torch.float32)
                )
                self.params_nld = {"alpha": self.alpha, "g_pre": g_pre_tensor}

                def nld(x):
                    g_pre = self.params["g_pre"]
                    g_post = 1.0 / g_pre
                    y = g_pre * x.clone().to(x.device)
                    y = ((1 - self.alpha) * y) / (1 - self.alpha * y)
                    y = y * g_post
                    y = torch.where(y > 1, 1, y)
                    return y

                self.nld = nld

            else:
                print("NLD is identity - You don't specified the nonlinearity")
                raise NotImplementedError(
                    "NLD is identity - You don't specified the nonlinearity"
                )
                self.nld = lambda x: x
                self.params_nld = {}

            print(f"{op_hp.NLD} NLD was chosen")
            self.params = {**self.params_nld}
            print(self.params)

    def update_params(self, params):
        pass

    def state_dict(self):
        # Call the parent class state_dict method
        # state_dict = self.state_dict()
        # For example, add custom entries:

        state_dict = {}
        state_dict["op_hp"] = self.op_hp
        state_dict["sample_rate"] = self.sample_rate
        print(self.params)
        state_dict["params"] = self.params.copy()

        return state_dict

    def load_checkpoint(self, checkpoint):
        state_dict_data = torch.load(checkpoint, map_location=torch.device("cpu"))
        self.__init__(state_dict_data["op_hp"], state_dict_data["sample_rate"])
        # remove the custom entries
        self.params = state_dict_data.pop("params")

    def adapt_params_to_SDR(self, ref, SDRtarget):
        if self.op_hp.NLD == "hardclip_symmetrical":
            x = ref.cpu().numpy()
            nld = lambda x, limit: np.clip(x, -limit, limit)

            def find_clip_value(thresh):
                print("thresh", thresh)
                xclipped = nld(x, thresh)
                sdr = 20 * np.log10(
                    np.linalg.norm(x) / (np.linalg.norm(x - xclipped) + 1e-7)
                )
                print("sdr", sdr, SDRtarget)
                return np.abs(sdr - SDRtarget)

            clip_value = fsolve(find_clip_value, 0.01)

            self.params["limit"].data = torch.tensor(
                clip_value, device=self.device, dtype=torch.float32
            )
            print("Adapted limit to SDR", self.params["limit"].data)

        elif self.op_hp.NLD == "softclip":

            x = ref.cpu().numpy()

            def nld(x, g_pre):
                g_post = 1.0 / g_pre
                return np.tanh(g_pre * x) * g_post

            def find_clip_value(thresh):
                print("thresh", thresh)
                xclipped = nld(x, thresh)
                sdr = 20 * np.log10(
                    np.linalg.norm(x) / (np.linalg.norm(x - xclipped) + 1e-7)
                )
                print("sdr", sdr, SDRtarget)
                return np.abs(sdr - SDRtarget)

            init_value = self.params["g_pre"].data.cpu().numpy()
            clip_value = fsolve(find_clip_value, init_value)
            self.params["g_pre"].data = torch.tensor(
                clip_value, device=self.device, dtype=torch.float32
            )
            print("Adapted g_pre to SDR", self.params["g_pre"].data)

        elif self.op_hp.NLD == "HWR":
            pass
        elif self.op_hp.NLD == "quantize":
            pass
        elif self.op_hp.NLD == "foldback":
            x = ref.cpu().numpy()

            def nld(x, limit):
                self.params["limit"].data = torch.tensor(
                    limit, device=self.device, dtype=torch.float32
                )
                y = self.nld(torch.tensor(x, device=self.device, dtype=torch.float32))

                return y.cpu().numpy()

            def find_clip_value(thresh):
                print("thresh", thresh)
                xclipped = nld(x, thresh)
                sdr = 20 * np.log10(
                    np.linalg.norm(x) / (np.linalg.norm(x - xclipped) + 1e-7)
                )
                print("sdr", sdr, SDRtarget)
                return np.abs(sdr - SDRtarget)

            init_value = self.params["limit"].data.cpu().numpy()
            clip_value = fsolve(find_clip_value, init_value)
            self.params["limit"].data = torch.tensor(
                clip_value, device=self.device, dtype=torch.float32
            )
            print("Adapted limit to SDR", self.params["limit"].data)

        else:
            raise NotImplementedError(
                "Adaptation of parameters to SDR not implemented for NLD {}".format(
                    self.op_hp.NLD
                )
            )

    def initialize_coeffs(self, ref):
        if self.op_hp.NLD == "CCR":
            print("initializing CCR coefficients")
            coefs_grid = self.grid
            from utils.helper import find_closest

            self.params["spline_coefs"].data = find_closest(coefs_grid.view(-1), ref)
        else:
            print("No coefficients to initialize")
            pass

    def apply_stft(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        elif len(x.shape) == 2:
            pass
        else:
            raise ValueError("x must have shape (batch, samples) or (samples)")

        X = self.stft(x) / torch.sqrt(torch.sum(self.window**2))
        return X

    def apply_istft(self, X, length=None):
        if length is None:
            print("Warning: length is None, istft may crash")
            length_param = None
        else:
            length_param = length

        X *= torch.sqrt(torch.sum(self.window**2))
        x = self.istft(X, length=length_param)
        return x

    def istft(self, X, length=None):
        return torch.istft(
            X,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            onesided=True,
            center=True,
            normalized=False,
            return_complex=False,
            length=length,
        )

    def stft(self, x):
        return torch.stft(
            x,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            onesided=True,
            return_complex=True,
            normalized=False,
            pad_mode="constant",
        )

    def save_checkpoint(self, path):
        torch.save(self.params, path)

    def degradation(self, x):
        y = self.nld(x)
        return y
