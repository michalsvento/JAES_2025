from tqdm import tqdm
import torch
from utils.losses import get_loss
from testing.EulerHeunSampler import EulerHeunSampler
from torch.utils.checkpoint import checkpoint


class EulerHeunSamplerDPS(EulerHeunSampler):
    """
    Euler Heun sampler for DPS
    inverse problem solver
    """

    def __init__(self, model, diff_params, args):
        super().__init__(model, diff_params, args)

        self.zeta = self.args.tester.posterior_sampling.zeta

        self.sampler_unconditional = EulerHeunSampler(model, diff_params, args)

    def predict_unconditional(self, shape, device):
        return self.sampler_unconditional.predict_unconditional(shape, device)

    def predict_conditional(
        self,
        y,  # observations
        operator_blind=None,  # degradation operator (assuming we define it in the tester)
        blind=False,
        reference=None,  # GT reference (just for logging)
        init_callback=None,
        logging_callback=None,
        operator_ref=None,
        evaluation_callback=None,
        save_path=None,
        **kwargs,
    ):
        """
        Perform conditional sampling
        input:
            y: observations torch.Tensor with shape (B, T)
            operator_blind: optimizable (blind) forward operator (assuming we define it in the tester)
            blind: boolean, if True, we optimize the operator parameters
            reference: GT reference (just for logging) torch.Tensor with shape (B, T)
            logging_callback: lambda function to log the results
            operator_ref: reference operator (used as forward operator in informed case)
            evaluate_callback: function to evaluate the results
            save_path: path to save the results

        """

        self.evaluation_callback = evaluation_callback
        self.logging_callback = logging_callback
        self.save_path = save_path
        self.y = y
        self.reference = reference

        if blind:
            self.operator = operator_blind
            self.operator_ref = operator_ref
        else:
            self.operator = operator_ref
            self.operator_ref = operator_ref

        self.rec_loss = get_loss(
            self.args.tester.posterior_sampling.rec_loss, operator=self.operator
        )

        if blind:
            try:
                self.optimizer_operator = torch.optim.Adam(
                    self.operator.params.values(),
                    lr=self.args.tester.posterior_sampling.blind_hp.lr_op,
                    weight_decay=self.args.tester.posterior_sampling.blind_hp.weight_decay,
                    betas=(
                        self.args.tester.posterior_sampling.blind_hp.beta1,
                        self.args.tester.posterior_sampling.blind_hp.beta2,
                    ),
                )
            except:
                self.optimizer_operator = torch.optim.Adam(
                    self.operator.params,
                    lr=self.args.tester.posterior_sampling.blind_hp.lr_op,
                    weight_decay=self.args.tester.posterior_sampling.blind_hp.weight_decay,
                    betas=(
                        self.args.tester.posterior_sampling.blind_hp.beta1,
                        self.args.tester.posterior_sampling.blind_hp.beta2,
                    ),
                )

            self.rec_loss_params = get_loss(
                self.args.tester.posterior_sampling.rec_loss_params,
                operator=self.operator,
            )

            try:
                if self.args.tester.posterior_sampling.imp_response_reg.use:
                    self.imp_response_reg = get_loss(
                        self.args.tester.posterior_sampling.imp_response_reg,
                        operator=self.operator,
                    )
                else:
                    print("no imp response reg ")
                    self.imp_response_reg = None
            except:
                print("no imp response reg (not defined)")
                self.imp_response_reg = None
        else:
            pass

        if self.args.tester.wandb.use:
            self.setup_wandb()
            dict = {"y": self.y, "reference": self.reference}
            init_callback(
                wandb_run=self.wandb_run, dict=dict, device=self.y.device, step=0
            )
        else:
            self.wandb_run = None

        shape = self.y.shape

        return self.predict(shape, self.y.device, blind)

    def get_likelihood_score(self, x_den, x, t):

        torch.cuda.empty_cache()
        y = self.y
        y_hat = self.operator.degradation(x_den)
        rec = self.rec_loss(y, y_hat)
        loss = rec
        loss.backward(retain_graph=False)
        rec_grads = x.grad

        if self.args.tester.posterior_sampling.grad_clip is not None:
            rec_grads = torch.clamp(
                rec_grads,
                -self.args.tester.posterior_sampling.grad_clip,
                self.args.tester.posterior_sampling.grad_clip,
            )

        if self.args.tester.posterior_sampling.normalization_type == "grad_norm":
            normguide = torch.norm(rec_grads) / ((x.shape[0] * x.shape[-1]) ** 0.5)
            zeta = self.zeta / (normguide + 1e-8)
            return -zeta * rec_grads / t, rec

        elif self.args.tester.posterior_sampling.normalization_type == "loss_norm":
            normguide = rec / ((x.shape[0] * x.shape[-1]) ** 0.5)
            zeta = self.zeta / (normguide + 1e-8)
            return -zeta * rec_grads / t, rec

        else:
            raise NotImplementedError(
                f"normalization type {self.args.tester.posterior_sampling.normalization_type} not implemented"
            )

    def optimize_op(self, x_den):
        """
        Optimize the operator parameters
        """

        rec_loss = torch.tensor([0])
        for i in tqdm(
            range(self.args.tester.posterior_sampling.blind_hp.op_updates_per_step)
        ):

            if isinstance(self.operator.params, list):
                for k in range(len(self.operator.params)):
                    self.operator.params[k].requires_grad = True
            elif isinstance(self.operator.params, dict):
                for p in self.operator.params.values():
                    p.requires_grad = True
            else:
                self.operator.params.requires_grad = True

            self.optimizer_operator.zero_grad()

            y_ref = self.y
            y_hat = self.operator.degradation(x_den.to(self.y.device))

            print("y_ref device", y_ref.device, "y_hat device", y_hat.device)

            if self.rec_loss_params is not None:
                rec_loss = self.rec_loss_params(y_ref, y_hat)
                loss = rec_loss
                assert torch.isnan(rec_loss).any() == False, f"rec_loss is Nan"
            else:
                raise NotImplementedError("rec_loss_params is not implemented")
                loss = 0.0

            loss = rec_loss
            loss.backward()

            if self.args.tester.posterior_sampling.blind_hp.grad_clip_use:
                for name, param in self.operator.params.items():
                    if param.grad is not None:
                        print(f"{name} grad norm before clipping: {param.grad.norm()}")
                    else:
                        print(f"{name} has no gradients")

                # #gradient clipping
                max_norm = (
                    self.args.tester.posterior_sampling.blind_hp.grad_clip
                )  # Set the maximum norm for gradient clipping

                if isinstance(self.operator.params, list):
                    for k in range(len(self.operator.params)):
                        if (
                            self.operator.params[k].grad is not None
                        ):  # Ensure there's a gradient
                            torch.nn.utils.clip_grad_norm_(
                                self.operator.params[k], max_norm
                            )
                elif isinstance(self.operator.params, dict):
                    for p in self.operator.params.values():
                        if p.grad is not None:  # Ensure there's a gradient
                            torch.nn.utils.clip_grad_norm_(p, max_norm)
                else:
                    if (
                        self.operator.params.grad is not None
                    ):  # Ensure there's a gradient
                        torch.nn.utils.clip_grad_norm_(self.operator.params, max_norm)

                for name, param in self.operator.params.items():
                    if param.grad is not None:
                        print(f"{name} grad norm after clipping: {param.grad.norm()}")
                    else:
                        print(f"{name} has no gradients")

            self.optimizer_operator.step()

    def free_grads(self):
        try:
            # free all the gradients of the operator
            for p in self.operator.params.values():
                p.grad = None
        except:
            pass

    def step(self, x_i, t_i, t_iplus1, gamma_i, blind=False):

        x_hat, t_hat = self.stochastic_timestep(x_i, t_i, gamma_i)
        x_hat.requires_grad = True
        x_den = self.get_Tweedie_estimate(x_hat, t_hat)

        if self.args.tester.posterior_sampling.constraint_magnitude.use:
            print("constraint magnitude")
            if (
                t_i
                >= self.args.tester.posterior_sampling.constraint_magnitude.stop_constraint_time
            ):
                if (
                    self.args.tester.posterior_sampling.constraint_magnitude.strategy
                    == "sigma_data"
                ):
                    x_den = (
                        self.args.tester.posterior_sampling.constraint_magnitude.speech_scaling
                        / x_den.detach().std()
                        * x_den
                    )
                elif (
                    self.args.tester.posterior_sampling.constraint_magnitude.strategy
                    == "oracle"
                ):
                    x_den = self.reference.std() / x_den.detach().std() * x_den
                else:
                    raise NotImplementedError(
                        f"strategy {self.args.tester.posterior_sampling.constraint_magnitude.strategy} not implemented"
                    )

        print("get likelihood score")
        lh_score, rec_loss_value = self.get_likelihood_score(x_den, x_hat, t_hat)
        x_hat = x_hat.detach()
        x_den = x_den.detach()

        if blind:
            self.optimize_op(x_den.clone().detach())

        score = self.Tweedie2score(x_den, x_hat, t_hat)
        ode_integrand = self.diff_params._ode_integrand(x_hat, t_hat, score + lh_score)
        dt = t_iplus1 - t_hat

        if t_iplus1 != 0 and self.order == 2:
            # second order correction
            t_prime = t_iplus1
            x_prime = x_hat + dt * ode_integrand
            x_prime.requires_grad_(True)
            x_den = self.get_Tweedie_estimate(x_prime, t_prime)

            if self.args.tester.posterior_sampling.constraint_magnitude.use:
                if (
                    t_i
                    >= self.args.tester.posterior_sampling.constraint_magnitude.stop_constraint_time
                ):
                    if (
                        self.args.tester.posterior_sampling.constraint_magnitude.strategy
                        == "sigma_data"
                    ):
                        x_den = (
                            self.args.tester.posterior_sampling.constraint_magnitude.speech_scaling
                            / x_den.detach().std()
                            * x_den
                        )
                    elif (
                        self.args.tester.posterior_sampling.constraint_magnitude.strategy
                        == "oracle"
                    ):
                        x_den = self.reference.std() / x_den.detach().std() * x_den
                    else:
                        raise NotImplementedError(
                            f"strategy {self.args.tester.posterior_sampling.constraint_magnitude.strategy} not implemented"
                        )

            if blind:
                self.optimize_op(x_den.clone().detach())

            lh_score_next, rec_loss_value = self.get_likelihood_score(
                x_den, x_prime, t_prime
            )
            x_prime.detach_()

            score = self.Tweedie2score(x_den, x_prime, t_prime)

            ode_integrand_next = self.diff_params._ode_integrand(
                x_prime, t_prime, score + lh_score_next
            )
            ode_integrand_midpoint = 0.5 * (ode_integrand + ode_integrand_next)
            x_iplus1 = x_hat + dt * ode_integrand_midpoint

        else:

            x_iplus1 = x_hat + dt * ode_integrand

        if self.args.tester.wandb.use:
            with torch.no_grad():
                logging_dictionary = {
                    "rec_loss_value": rec_loss_value,
                    "lh_score_norm": torch.norm(lh_score),
                    "score_norm": torch.norm(score),
                    "t_i": t_hat,
                    "t_iplus1": t_iplus1,
                }
                self.logging_callback(
                    wandb_run=self.wandb_run,
                    dict=logging_dictionary,
                    step_counter=self.step_counter,
                )

        return x_iplus1.detach_(), x_den.detach()

    def initialize_x(self, shape, t_i, device):

        if self.args.tester.posterior_sampling.initialization.mode == "y+noise":
            if self.args.tester.posterior_sampling.initialization.scale_y is not None:
                scale_y = (
                    self.args.tester.posterior_sampling.initialization.scale_y
                    / self.y.std()
                )
            else:
                scale_y = 1.0
            x = t_i * torch.randn(shape).to(device) + scale_y * self.y
        elif self.args.tester.posterior_sampling.initialization.mode == "noise":
            x = t_i * torch.randn(shape).to(device)
        return x

    def predict(
        self,
        shape,  # observations (lowpssed signal) Tensor with shape ??
        device,  # lambda function
        blind=False,
    ):

        # get the noise schedule
        t = self.create_schedule().to(device)

        # sample prior
        x = self.initialize_x(shape, t[0], device)

        # parameter for langevin stochasticity, if Schurn is 0, gamma will be 0 to, so the sampler will be deterministic
        gamma = self.get_gamma(t).to(device)

        self.step_counter = 0

        with torch.no_grad():
            dict = {
                "x_den": None,
                "y": self.y,
                "reference": self.reference,
                "x": x,
                "operator": self.operator,
                "operator_ref": self.operator_ref,
                "reference": self.reference,
            }
            if self.evaluation_callback is not None:
                self.evaluation_callback(
                    wandb_run=self.wandb_run,
                    dict=dict,
                    device=x.device,
                    step=self.step_counter,
                )

            try:
                path = self.save_path + f"/checkpoint_{i}.pt"
                print("saving checkpoint at ", path)
                self.operator.save_checkpoint(path)
            except:
                print("could not save checkpoint")
                pass

        for i in tqdm(range(0, self.T, 1)):
            self.step_counter = i
            print("step", i)
            x, x_den = self.step(x, t[i], t[i + 1], gamma[i], blind)

            if i % self.args.tester.set.evaluate_every == 1:

                with torch.no_grad():
                    dict = {
                        "x_den": x_den,
                        "y": self.y,
                        "reference": self.reference,
                        "x": x,
                        "operator": self.operator,
                        "operator_ref": self.operator_ref,
                        "reference": self.reference,
                    }
                    if self.evaluation_callback is not None:
                        self.evaluation_callback(
                            wandb_run=self.wandb_run,
                            dict=dict,
                            device=x.device,
                            step=self.step_counter,
                        )

                    try:
                        path = self.save_path + f"/checkpoint_{i}.pt"
                        print("saving checkpoint at ", path)
                        self.operator.save_checkpoint(path)
                    except:
                        print("could not save checkpoint")
                        pass

        with torch.no_grad():
            dict = {
                "x_den": x_den,
                "y": self.y,
                "reference": self.reference,
                "x": x,
                "operator": self.operator,
                "operator_ref": self.operator_ref,
                "reference": self.reference,
            }
            if self.evaluation_callback is not None:
                self.evaluation_callback(
                    wandb_run=self.wandb_run,
                    dict=dict,
                    device=x.device,
                    step=self.step_counter,
                )

            try:
                path = self.save_path + f"/checkpoint_{i}.pt"
                print("saving checkpoint at ", path)
                self.operator.save_checkpoint(path)
            except:
                print("could not save checkpoint")
                pass

        if self.args.tester.wandb.use:
            self.wandb_run.finish()

        return x_den.detach()

    def get_Tweedie_estimate(self, x, t_i):
        if self.args.tester.set.minibatch_size_diffusion == -1:
            x_hat = self.diff_params.denoiser(x.unsqueeze(1), self.model, t_i).squeeze(
                1
            )
        else:
            # separate processing in minibatches
            orig_shape = x.shape
            x = x.view(-1, self.args.tester.set.minibatch_size_diffusion, x.shape[-1])
            for k in range(x.shape[0]):

                def call_denoiser(x_k):
                    x_den = self.diff_params.denoiser(
                        x_k.unsqueeze(1), self.model, t_i
                    ).squeeze(1)
                    return x_den

                x_hat = checkpoint(call_denoiser, x[k], use_reentrant=True)

                if k == 0:
                    x_hat_batch = x_hat.unsqueeze(0)
                else:
                    x_hat_batch = torch.cat((x_hat_batch, x_hat.unsqueeze(0)), dim=0)

            x_hat = x_hat_batch.view(orig_shape)

        return x_hat
