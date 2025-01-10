import torch.nn as nn
import torch
import abc


### Abstract class


class Operator(nn.Module):

    @abc.abstractmethod
    def degradation(self, *args, **kwargs):
        """
        Forward Pass for degradation with given parameters
        """
        pass

    @abc.abstractmethod
    def update_params(self, *args, **kwargs):
        """
        Method for updating parameters in blind scenarios or when loading new settings with same class
        """
        pass

    def prepare_optimization(self, x_den, y):
        """
        Some preprocessing for optimizing the parameters. Empty by default
        """
        return x_den, y

    def constrain_params(self):
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

    def adapt_params_to_SDR(self, ref, sdrtarget):
        pass

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

    def initialize_coeffs(self, y):
        pass
