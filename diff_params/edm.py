import torch
import numpy as np

import utils.training_utils as utils
from diff_params.shared import SDE


class EDM(SDE):
    """
    Definition of the diffusion parameterization, following ( Karras et al., "Elucidating...", 2022).
    This includes only the utilities needed for training, not for sampling.
    """

    def __init__(self, type, sde_hp):

        super().__init__(type, sde_hp)

        self.sigma_data = (
            self.sde_hp.sigma_data
        )  # depends on the training data!! precalculated variance of the dataset
        self.sigma_min = self.sde_hp.sigma_min
        self.sigma_max = self.sde_hp.sigma_max
        self.rho = self.sde_hp.rho

    def sample_time_training(self, N):
        """
        For training, getting t according to a similar criteria as sampling. Simpler and safer to what Karras et al. did
        Args:
            N (int): batch size
        """
        a = torch.rand(N)
        t = (
            self.sigma_max ** (1 / self.rho)
            + a * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho

        return t

    def sample_prior(self, shape=None, t=None):
        """
        Just sample some gaussian noise, nothing more
        Args:
            shape (tuple): shape of the noise to sample, something like (B,T)
        """
        assert shape is not None
        if t is not None:
            n = torch.randn(shape).to(t.device) * t
        else:
            n = torch.randn(shape)
        return n

    def cskip(self, sigma):
        """
        Just one of the preconditioning parameters
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)

        """
        return self.sigma_data**2 * (sigma**2 + self.sigma_data**2) ** -1

    def cout(self, sigma):
        """
        Just one of the preconditioning parameters
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        return sigma * self.sigma_data * (self.sigma_data**2 + sigma**2) ** (-0.5)

    def cin(self, sigma):
        """
        Just one of the preconditioning parameters
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        return (self.sigma_data**2 + sigma**2) ** (-0.5)

    def cnoise(self, sigma):
        """
        preconditioning of the noise embedding
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        return (1 / 4) * torch.log(sigma)

    def lambda_w(self, sigma):
        """
        Score matching loss weighting
        """
        return (sigma * self.sigma_data) ** (-2) * (self.sigma_data**2 + sigma**2)

    def Tweedie2score(self, tweedie, xt, t, *args, **kwargs):
        return (tweedie - self._mean(xt, t)) / self._std(t) ** 2

    def score2Tweedie(self, score, xt, t, *args, **kwargs):
        return self._std(t) ** 2 * score + self._mean(xt, t)

    def _mean(self, x, t):
        return x

    def _std(self, t):
        return t

    def _ode_integrand(self, x, t, score):
        return -t * score

    def _corrector_(self, x, score, gamma, t):
        w = torch.randn_like(x)
        # annealed langevin dynamics
        step_size = 0.5 * (gamma * t) ** 2

        return x + step_size * score + torch.sqrt(2 * step_size) * w

    def denoiser(self, xn, net, t, *args, **kwargs):
        """
        This method does the whole denoising step, which implies applying the model and the preconditioning
        Args:
            x (Tensor): shape: (B,1,T) Intermediate noisy latent to denoise
            model (nn.Module): Model of the denoiser
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        sigma = self._std(t).unsqueeze(-1)
        sigma = sigma.view(*sigma.size(), *(1,) * (xn.ndim - sigma.ndim))

        cskip = self.cskip(sigma)
        cout = self.cout(sigma)
        cin = self.cin(sigma)
        cnoise = self.cnoise(sigma.squeeze())

        # check if cnoise is a scalar, if so, repeat it
        if len(cnoise.shape) == 0:
            cnoise = cnoise.repeat(
                xn.shape[0],
            ).unsqueeze(-1)
        else:
            cnoise = cnoise.view(
                xn.shape[0],
            ).unsqueeze(-1)

        # print(xn.shape, cskip.shape, cout.shape, cin.shape, cnoise.shape)
        print("xn", xn.shape)
        x_hat = cskip * xn + cout * net(
            (cin * xn).to(torch.float32), cnoise.to(torch.float32)
        ).to(
            xn.dtype
        )  # this will crash because of broadcasting problems, debug later!

        try:
            if net.CQTransform is not None:
                x_hat = net.CQTransform.apply_hpf_DC(x_hat)
        except:
            pass

        return x_hat


class EDM_whitened(EDM):
    """
    Definition of the diffusion parameterization, following ( Karras et al., "Elucid
    """

    def __init__(
        self,
        file_state_dict,
        type,
        use_speaker_cond,
        p_drop_speaker,
        sde_hp,
        speaker_cond_sample_rate=None,
        our_sample_rate=None,
    ):

        super().__init__(
            type,
            use_speaker_cond,
            p_drop_speaker,
            sde_hp,
            speaker_cond_sample_rate,
            our_sample_rate,
        )

        self.whitener_dict = torch.load(file_state_dict)

    def loss_fn(self, net, x, n=None, *args, **kwargs):
        """
        Loss function, which is the mean squared error between the denoised latent and the clean latent
        Args:
            net (nn.Module): Model of the denoiser
            x (Tensor): shape: (B,T) Intermediate noisy latent to denoise
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        t = self.sample_time_training(x.shape[0]).to(x.device)

        if self.use_speaker_cond:
            x_4_emb = x.detach().clone().to(x.device)
            if self.speaker_cond_sample_rate != self.our_sample_rate:
                # convert from 44100 to 16000
                x_4_emb = torchaudio.functional.resample(
                    x_4_emb, self.speaker_cond_sample_rate, self.our_sample_rate
                )

            self.classifier.device = x.device
            self.classifier.mods.embedding_model.to(x.device)
            embeddings = self.classifier.encode_batch(x_4_emb)
            # print(embeddings.device)
            embeddings = embeddings.to(x.device)
            # apply dropout (p_drop_speaker) to the embeddings
            if self.p_drop_speaker > 0:
                to_drop = torch.rand(embeddings.shape[0]) < self.p_drop_speaker
                embeddings[to_drop] *= 0

        x_white = utils.whitening(x, self.whitener_dict)

        input, target, cnoise = self.prepare_train_preconditioning(x_white, t, n=n)

        # print("embeddings", embeddings, to_drop)
        # print(torch.isnan(input).any(), "nan in input")
        # print(torch.isnan(target).any(), "nan in target")
        # print(torch.isnan(cnoise).any(), "nan in cnoise")
        # print(torch.isnan(embeddings).any(), "nan in embeddings")

        estimate = net(
            input.unsqueeze(1), cnoise, speaker_emb=embeddings.squeeze(1)
        ).squeeze(1)

        # print(torch.isnan(estimate).any(), "nan in estimate")
        error = estimate - target

        # apply this on the trainer
        # try:
        #    #this will only happen if the model is cqt-based, if it crashes it is normal
        #    if self.args.net.use_cqt_DC_correction:
        #        error=net.CQTransform.apply_hpf_DC(error) #apply the DC correction to the error as we dont want to propagate the DC component of the error as the network is discarding it. It also applies for the nyquit frequency, but this is less critical.
        # except:
        #    pass

        # here we have the chance to apply further emphasis to the error, as some kind of perceptual frequency weighting could be
        return error**2, self._std(t)
