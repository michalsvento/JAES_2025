import torch
import numpy as np
import abc

import utils.training_utils as utils


class SDE:
    """
    Definition of the diffusion following the parameterization as in ( Karras et al., "Elucidating...", 2022).
    This includes only the utilities needed for training, not for sampling.
    """

    def __init__(self, type, sde_hp):

        self.type = type
        self.sde_hp = sde_hp

    @abc.abstractmethod
    def sample_time_training(self, N):
        """
        For training, getting t according to a similar criteria as sampling.
        Args:
            N (int): batch size
        """
        pass

    @abc.abstractmethod
    def sample_prior(self, shape, *args, **kwargs):
        """
        Just sample some gaussian noise, nothing more
        Args:
            shape (tuple): shape of the noise to sample, something like (B,T)
        """
        pass

    @abc.abstractmethod
    def cskip(self, sigma, *args, **kwargs):
        """
        Just one of the preconditioning parameters
        """
        pass

    @abc.abstractmethod
    def cout(self, sigma, *args, **kwargs):
        """
        Just one of the preconditioning parameters
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        pass

    @abc.abstractmethod
    def cin(self, sigma, *args, **kwargs):
        """
        Just one of the preconditioning parameters
        """
        pass

    @abc.abstractmethod
    def cnoise(self, sigma, *args, **kwargs):
        """
        preconditioning of the noise embedding
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        pass

    @abc.abstractmethod
    def lambda_w(self, sigma, *args, **kwargs):
        """
        Score matching loss weighting
        """
        pass

    @abc.abstractmethod
    def _mean(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def _std(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def _ode_integrand(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def Tweedie2score(self, tweedie, xt, t, *args, **kwargs):
        pass

    @abc.abstractmethod
    def score2Tweedie(self, score, xt, t, *args, **kwargs):
        pass

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
            )
        else:
            cnoise = cnoise.view(
                xn.shape[0],
            )

        return cskip * xn + cout * net(
            cin * xn, cnoise
        )  # this will crash because of broadcasting problems, debug later!

    def prepare_train_preconditioning(self, x, t, n=None, *args, **kwargs):
        # weight=self.lambda_w(sigma)
        # Eloi: Is calling the denoiser here a good idea? Maybe it would be better to apply directly the preconditioning as in the paper, even though Karras et al seem to do it this way in their code
        # Jm: I don't mind that preconditioning form actually, makes the loss also more normalized and easier to compare in between runs/SDEs i.m.o.

        mu, sigma = self._mean(x, t), self._std(t).unsqueeze(-1)
        sigma = sigma.view(*sigma.size(), *(1,) * (x.ndim - sigma.ndim))
        if n is None:
            n = self.sample_prior(shape=x.shape).to(x.device)
        x_perturbed = mu + sigma * n
        # self.sample_prior(x.shape).to(x.device)

        cskip = self.cskip(sigma)
        cout = self.cout(sigma)
        cin = self.cin(sigma)
        cnoise = self.cnoise(sigma.squeeze())

        # check if cnoise is a scalar, if so, repeat it
        if len(cnoise.shape) == 0:
            cnoise = cnoise.repeat(
                x.shape[0],
            )
        else:
            cnoise = cnoise.view(
                x.shape[0],
            )

        target = 1 / cout * (x - cskip * x_perturbed)

        return cin * x_perturbed, target, cnoise

    def loss_fn(self, net, x, n=None, *args, **kwargs):
        """
        Loss function, which is the mean squared error between the denoised latent and the clean latent
        Args:
            net (nn.Module): Model of the denoiser
            x (Tensor): shape: (B,T) Intermediate noisy latent to denoise
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        t = self.sample_time_training(x.shape[0]).to(x.device)

        input, target, cnoise = self.prepare_train_preconditioning(x, t, n=n)

        # print("input",input.shape,"cnoise", cnoise.shape)

        if len(cnoise.shape) == 1:
            # dirty patch
            cnoise = cnoise.unsqueeze(-1)

        if input.ndim == 2:
            input = input.unsqueeze(1)

        estimate = net(input, cnoise)

        if target.ndim == 2 and estimate.ndim == 3:
            estimate = estimate.squeeze(1)
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
