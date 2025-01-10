from tqdm import tqdm
import torch
import wandb
from testing.Sampler import Sampler


class EulerHeunSampler(Sampler):

    def __init__(self, model, diff_params, args):
        print("diff_params", diff_params)
        print("args", args)
        super().__init__(model, diff_params, args)

        # stochasticity parameters
        self.Schurn = self.args.tester.sampling_params.Schurn
        self.Snoise = self.args.tester.sampling_params.Snoise
        self.Stmin = self.args.tester.sampling_params.Stmin
        self.Stmax = self.args.tester.sampling_params.Stmax

        # order of the sampler
        self.order = self.args.tester.sampling_params.order

    def predict_conditional(self, *args, **kwargs):
        raise NotImplementedError

    def predict_unconditional(
        self, shape, device  # observations (lowpssed signal) Tensor with shape ??
    ):
        self.y = None
        self.degradation = None
        if self.args.tester.wandb.use:
            self.setup_wandb()

        return self.predict(shape, device)

    def get_gamma(self, t):
        """
        Get the parameter gamma that defines the stochasticity of the sampler
        Args
            t (Tensor): shape: (N_steps, ) Tensor of timesteps, from which we will compute gamma
        """
        N = t.shape[0]
        gamma = torch.zeros(t.shape).to(t.device)

        # If desired, only apply stochasticity between a certain range of noises Stmin is 0 by default and Stmax is a huge number by default. (Unless these parameters are specified, this does nothing)
        indexes = torch.logical_and(t > self.Stmin, t < self.Stmax)

        # We use Schurn=5 as the default in our experiments
        gamma[indexes] = gamma[indexes] + torch.min(
            torch.Tensor([self.Schurn / N, 2 ** (1 / 2) - 1])
        )

        return gamma

    def stochastic_timestep(self, x, t, gamma, Snoise=1):
        t_hat = (
            t + gamma * t
        )  # if gamma_sig[i]==0 this is a deterministic step, make sure it doed not crash
        epsilon = (
            torch.randn(x.shape).to(x.device) * Snoise
        )  # sample Gaussiannoise, Snoise is 1 by default
        if t_hat <= t:
            x_hat = x
            print(f"t_hat<=t, gamma {gamma}")
        else:
            print(t_hat, t)
            x_hat = x + ((t_hat**2 - t**2) ** (1 / 2)) * epsilon  # Perturb data
        return x_hat, t_hat

    def step(self, x_i, t_i, t_iplus1, gamma_i, blind=False):

        with torch.no_grad():
            x_hat, t_hat = x_i, t_i
            x_den = self.get_Tweedie_estimate(x_hat, t_hat)
            score = self.Tweedie2score(x_den, x_hat, t_hat)
            ode_integrand = self.diff_params._ode_integrand(x_hat, t_hat, score)
            dt = t_iplus1 - t_hat

            if t_iplus1 != 0 and self.order == 2:  # second order correction
                t_prime = t_iplus1
                x_prime = x_hat + dt * ode_integrand
                x_den = self.get_Tweedie_estimate(x_prime, t_prime)
                score = self.Tweedie2score(x_den, x_prime, t_prime)
                ode_integrand_next = self.diff_params._ode_integrand(
                    x_prime, t_prime, score
                )
                ode_integrand_midpoint = 0.5 * (ode_integrand + ode_integrand_next)
                x_iplus1 = x_hat + dt * ode_integrand_midpoint

            else:
                x_iplus1 = x_hat + dt * ode_integrand

            return x_iplus1, x_den

    def predict(
        self,
        shape,  # observations (lowpssed signal) Tensor with shape ??
        device,  # lambda function
        blind=False,
    ):

        # get the noise schedule
        t = self.create_schedule().to(device).to(torch.float64)

        # sample prior
        x = (
            self.diff_params.sample_prior(t=t[0], shape=shape)
            .to(device)
            .to(torch.float64)
        )

        x_init = x.clone()

        # parameter for langevin stochasticity, if Schurn is 0, gamma will be 0 to, so the sampler will be deterministic
        gamma = self.get_gamma(t).to(device)

        for i in tqdm(range(0, self.T, 1)):
            self.step_counter = i
            print("step", i, x.shape, shape)
            x, x_den = self.step(x, t[i], t[i + 1], gamma[i], blind)
            print("Logging after step", x.std(), x_den.std(), t[i], t[i + 1], gamma[i])

            if self.args.tester.wandb.use:
                if i % self.args.tester.wandb.log_every == 0:
                    maxim = torch.max(torch.abs(x[0])).detach().cpu().numpy()
                    if maxim < 1:
                        maxim = 1
                    for k in range(0, len(x)):
                        self.wandb_run.log(
                            {
                                "x"
                                + str(k): wandb.Audio(
                                    x[k].detach().cpu().numpy() / maxim,
                                    sample_rate=self.args.exp.sample_rate,
                                )
                            },
                            step=i,
                        )
                    maxim = torch.max(torch.abs(x_den[0])).detach().cpu().numpy()
                    if maxim < 1:
                        maxim = 1
                    for k in range(0, len(x_den)):
                        self.wandb_run.log(
                            {
                                "x_tweedie"
                                + str(k): wandb.Audio(
                                    x_den[k].detach().cpu().numpy() / maxim,
                                    sample_rate=self.args.exp.sample_rate,
                                )
                            },
                            step=i,
                        )

        if self.args.tester.wandb.use:
            self.wandb_run.finish()

        return x_den.detach(), x_init.detach()
