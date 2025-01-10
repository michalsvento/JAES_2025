from datetime import date
from functools import partial
import re
import torch
import os
import wandb
import copy
from glob import glob
from tqdm import tqdm
import omegaconf
import hydra
import utils.log as utils_logging
import utils.training_utils as tr_utils
import utils.testing_utils as tt_utils


class Tester:
    def __init__(
        self,
        args,
        network,
        diff_params,
        inference_train_set=None,
        inference_test_set=None,
        device=None,
        in_training=False,
        training_wandb_run=None,
    ):
        self.args = args
        self.network = network
        self.diff_params = copy.copy(diff_params)
        self.device = device
        self.inference_train_set = inference_train_set
        self.inference_test_set = inference_test_set
        self.use_wandb = False  # hardcoded for now
        self.in_training = in_training
        self.sampler = hydra.utils.instantiate(
            args.tester.sampler, self.network, self.diff_params, self.args
        )

        if in_training:
            self.use_wandb = True
            # Will inherit wandb_run from Trainer
        else:  # If we use the tester in training, we will log in WandB in the Trainer() class, no need to create all these paths
            torch.backends.cudnn.benchmark = True
            if self.device is None:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )

    def setup_wandb(self):
        """
        Configure wandb, open a new run and log the configuration.
        """
        config = omegaconf.OmegaConf.to_container(
            self.args, resolve=True, throw_on_missing=True
        )
        self.wandb_run = wandb.init(
            project="testing" + self.args.exp.wandb.project,
            entity=self.args.exp.wandb.entity,
            config=config,
        )
        wandb.watch(self.network, log_freq=self.args.logging.heavy_log_interval)
        blind_estimation = (
            "BLIND"
            + self.args.tester.blind_distortion.op_type
            + "_"
            + self.args.tester.blind_distortion.op_hp.NLD
            + "_"
            + self.args.tester.blind_distortion.op_hp.filters
        )
        self.wandb_run.name = self.args.exp.exp_name + "_" + blind_estimation
        self.use_wandb = True

    def setup_wandb_run(self, run):
        # get the wandb run object from outside (in trainer.py or somewhere else)
        self.wandb_run = run
        self.use_wandb = True

    def load_latest_checkpoint(self):
        # load the latest checkpoint from self.args.model_dir
        try:
            # find latest checkpoint_id
            save_basename = f"{self.args.exp.exp_name}-*.pt"
            save_name = f"{self.args.model_dir}/{save_basename}"
            list_weights = glob(save_name)
            id_regex = re.compile(f"{self.args.exp.exp_name}-(\d*)\.pt")
            list_ids = [
                int(id_regex.search(weight_path).groups()[0])
                for weight_path in list_weights
            ]
            checkpoint_id = max(list_ids)

            state_dict = torch.load(
                f"{self.args.model_dir}/{self.args.exp.exp_name}-{checkpoint_id}.pt",
                map_location=self.device,
            )
            try:
                self.network.load_state_dict(state_dict["ema"])
            except Exception as e:
                print(e)
                print("Failed to load in strict mode, trying again without strict mode")
                self.network.load_state_dict(state_dict["model"], strict=False)

            print(f"Loaded checkpoint {checkpoint_id}")
            return True
        except (FileNotFoundError, ValueError):
            raise ValueError("No checkpoint found")

    def load_checkpoint(self, path):
        state_dict = torch.load(path, map_location=self.device, weights_only=False)
        try:
            self.it = state_dict["it"]
        except:
            self.it = 0

        print(f"loading checkpoint {self.it}")
        return tr_utils.load_state_dict(state_dict, ema=self.network)

    def log_audio(self, pred, name: str):
        if self.use_wandb:
            print(pred.shape)
            pred = pred.view(-1)
            maxim = torch.max(torch.abs(pred)).detach().cpu().numpy()
            if maxim < 1:
                maxim = 1
            self.wandb_run.log(
                {
                    name: wandb.Audio(
                        pred.detach().cpu().numpy() / maxim,
                        sample_rate=self.args.exp.sample_rate,
                    )
                },
                step=self.it,
            )

            if self.args.logging.log_spectrograms:
                raise NotImplementedError

    def test_conditional_supervised(self, mode):

        assert self.inference_test_set is not None, "No test set specified"

        x_all = []
        y_all = []
        preds_all = []
        for x, y in tqdm(self.inference_test_set):
            x = torch.tensor(x).float().to(self.device)
            y = torch.tensor(y).float().to(self.device)
            preds, noise_init = self.sampler.predict_conditional(y, self.device)
            x_all.append(x)
            y_all.append(y)
            preds_all.append(preds)

        x = torch.stack(x_all)
        y = torch.stack(y_all)
        preds = torch.stack(preds_all)
        sigma_data = self.args.tester.sampling_params.sde_hp.sigma_data

        if self.use_wandb:
            self.log_audio(x.view(-1), f"ref+{self.sampler.T}")  # Just log first sample
            self.log_audio(
                y.view(-1), f"distorted+{self.sampler.T}"
            )  # Just log first sample
            self.log_audio(
                preds.view(-1), f"preds+{self.sampler.T}"
            )  # Just log first sample
        else:
            if not self.in_training:
                for i in range(len(preds)):
                    path_generated = utils_logging.write_audio_file(
                        preds[i] * sigma_data,
                        self.args.exp.sample_rate,
                        f"unconditional_{self.args.tester.wandb.pair_id}",
                        path=self.paths["unconditional"],
                    )
                    path_generated_noise = utils_logging.write_audio_file(
                        noise_init[i],
                        self.args.exp.sample_rate,
                        f"noise_{self.args.tester.wandb.pair_id}",
                        path=self.paths["unconditional"],
                    )

        return x, y, preds

    def test_conditional_supervised_evaluation(self, mode):

        if self.inference_train_set is None:
            print("No inference train set specified")
            return
        if len(self.inference_train_set) == 0:
            print("No samples found in test set")
            return

        x_all = []
        y_all = []
        preds_all = []
        for i, data in enumerate(tqdm(self.inference_train_set)):
            self.prepare_directories(
                mode, unconditional=False, blind=True, string=str(i)
            )

            if (
                os.path.exists(os.path.join(self.paths[mode + "original"], "x.wav"))
                and not self.args.tester.set.overwrite
            ):
                continue

            if self.args.tester.distortion.target_type == "PairedDataset":
                original, distorted = data
                seg = torch.from_numpy(original).float().to(self.device)
                y = torch.from_numpy(distorted).float().to(self.device)
                assert (
                    seg.shape == y.shape
                ), "original and distorted should have the same shape"

            else:
                if self.args.tester.distortion.target_type == "WienerHammerstein":
                    from testing.operators.WienerHammerstein import WienerHammerstein

                    operator = WienerHammerstein(
                        self.args.tester.distortion.op_hp, self.args.exp.sample_rate
                    ).to(self.device)
                elif self.args.tester.distortion.target_type == "LPF":
                    from testing.operators.lowpass_filters import LPF

                    operator = LPF(
                        self.args.tester.distortion.op_hp, self.args.exp.sample_rate
                    ).to(self.device)
                elif self.args.tester.distortion.target_type == "Filter":
                    from testing.operators.filters import Filter

                    operator = Filter(
                        self.args.tester.distortion.op_hp, self.args.exp.sample_rate
                    ).to(self.device)
                elif self.args.tester.distortion.target_type == "Nonlinear":
                    from testing.operators.NonlinearModule import Nonlinear

                    operator = Nonlinear(
                        self.args.tester.distortion.op_hp, self.args.exp.sample_rate
                    ).to(self.device)
                elif self.args.tester.distortion.target_type == "Linear":
                    from testing.operators.LinearModule import Linear

                    operator = Linear(
                        self.args.tester.distortion.op_hp, self.args.exp.sample_rate
                    ).to(self.device)
                elif self.args.tester.blind_distortion.op_type == "LTI":
                    from testing.operators.LTIModule import LTI

                    operator = LTI(
                        self.args.tester.distortion.op_hp, self.args.exp.sample_rate
                    )

                with torch.no_grad():
                    # check if the data is a tuple or a single element
                    if isinstance(data, tuple):
                        if len(data) == 2:
                            original, _ = data
                        else:
                            raise ValueError("The tuple should have 2 elements")
                    else:
                        original = data

                    seg = torch.from_numpy(original).float().to(self.device)
                    if self.args.tester.set.normalize_rms is not None:
                        seg = self.args.tester.set.normalize_rms * seg / seg.std()

                    if self.args.tester.distortion.fix_SDR.use:
                        print(
                            "adapting to SDR"
                            + str(self.args.tester.distortion.fix_SDR.target_SDR)
                        )
                        operator.adapt_params_to_SDR(
                            seg, self.args.tester.distortion.fix_SDR.target_SDR
                        )

                    y = operator.degradation(seg)

            preds, noise_init = self.sampler.predict_conditional(y, self.device)

            path_generated = utils_logging.write_audio_file(
                preds,
                self.args.exp.sample_rate,
                f"x_hat_49",
                path=self.paths[mode + "reconstructed"],
            )

            path_generated = utils_logging.write_audio_file(
                y, self.args.exp.sample_rate, f"y", path=self.paths[mode + "degraded"]
            )

            path_generated = utils_logging.write_audio_file(
                seg, self.args.exp.sample_rate, f"x", path=self.paths[mode + "original"]
            )

    # ------------- UNCONDITIONAL SAMPLING ---------------#

    ##############################
    ### UNCONDITIONAL SAMPLING ###
    ##############################

    def sample_unconditional(self, mode):
        # the audio length is specified in the args.exp, doesnt depend on the tester --> well should probably change that
        audio_len = (
            self.args.exp.audio_len
            if not "audio_len" in self.args.tester.unconditional.keys()
            else self.args.tester.unconditional.audio_len
        )
        shape = [self.args.tester.unconditional.num_samples, audio_len]
        preds, noise_init = self.sampler.predict_unconditional(shape, self.device)
        sigma_data = self.args.tester.sampling_params.sde_hp.sigma_data

        if self.use_wandb:
            # preds=preds/torch.max(torch.abs(preds))
            self.log_audio(
                preds[0], f"unconditional+{self.sampler.T}"
            )  # Just log first sample
            # self.log_unconditional_metrics(preds) #But compute metrics on several
        else:
            try:
                if not self.in_training:
                    for i in range(len(preds)):
                        path_generated = utils_logging.write_audio_file(
                            preds[i] * sigma_data,
                            self.args.exp.sample_rate,
                            f"unconditional_{self.args.tester.wandb.pair_id}",
                            path=self.paths["unconditional"],
                        )
                        path_generated_noise = utils_logging.write_audio_file(
                            noise_init[i],
                            self.args.exp.sample_rate,
                            f"noise_{self.args.tester.wandb.pair_id}",
                            path=self.paths["unconditional"],
                        )
            except:
                pass

        return preds

    # ------------- CONDITIONAL SAMPLING ---------------#

    def test(self, mode, matched=False, blind=False):

        if self.inference_train_set is None:
            print("No inference train set specified")
            return
        if len(self.inference_train_set) == 0:
            print("No samples found in test set")
            return

        is_test_set = True
        if self.inference_test_set is None:
            print("No inference test set specified")
            print("not possible to run evaluation")
            is_test_set = False
        elif len(self.inference_test_set) == 0:
            print("No samples found in test set")
            is_test_set = False

        file_ids, degradeds, originals, preds = [], [], [], []

        print("Files will be saved in: ", self.paths[mode])

        xs = []
        ys = []

        # Setting of the known distortion operator
        # Load the previously genereated degraded samples - we dont know the original system at this point
        # We will degrade the signal with the known operator and then try to reconstruct it

        if self.args.tester.distortion.target_type == "PairedDataset":

            assert blind == False, "informed distortion not supported for PairedDataset"

            operator = None

            for i, (original, distorted) in enumerate(tqdm(self.inference_train_set)):
                seg = torch.from_numpy(original).float().to(self.device)
                y = torch.from_numpy(distorted).float().to(self.device)

                xs.append(seg)
                ys.append(y)

        else:
            if self.args.tester.distortion.target_type == "WienerHammerstein":
                from testing.operators.WienerHammerstein import WienerHammerstein

                operator = WienerHammerstein(
                    self.args.tester.distortion.op_hp, self.args.exp.sample_rate
                ).to(self.device)
            elif self.args.tester.distortion.target_type == "LPF":
                from testing.operators.lowpass_filters import LPF

                operator = LPF(
                    self.args.tester.distortion.op_hp, self.args.exp.sample_rate
                ).to(self.device)
            elif self.args.tester.distortion.target_type == "Filter":
                from testing.operators.filters import Filter

                operator = Filter(
                    self.args.tester.distortion.op_hp, self.args.exp.sample_rate
                ).to(self.device)
            elif self.args.tester.distortion.target_type == "Nonlinear":
                from testing.operators.NonlinearModule import Nonlinear

                operator = Nonlinear(
                    self.args.tester.distortion.op_hp, self.args.exp.sample_rate
                ).to(self.device)
            elif self.args.tester.distortion.target_type == "Linear":
                from testing.operators.LinearModule import Linear

                operator = Linear(
                    self.args.tester.distortion.op_hp, self.args.exp.sample_rate
                ).to(self.device)
            elif self.args.tester.blind_distortion.op_type == "LTI":
                from testing.operators.LTIModule import LTI

                operator = LTI(
                    self.args.tester.distortion.op_hp, self.args.exp.sample_rate
                )
            else:
                raise NotImplementedError(
                    f"Operator {self.args.tester.distortion.target_type} not implemented"
                )

            with torch.no_grad():
                for i, data in enumerate(tqdm(self.inference_train_set)):

                    # check if the data is a tuple or a single element
                    if isinstance(data, tuple):
                        if len(data) == 2:
                            original, _ = data
                        else:
                            raise ValueError("The tuple should have 2 elements")
                    else:
                        original = data

                    seg = torch.from_numpy(original).float().to(self.device)
                    if self.args.tester.set.normalize_rms is not None:
                        seg = self.args.tester.set.normalize_rms * seg / seg.std()

                    if self.args.tester.distortion.fix_SDR.use:
                        operator.adapt_params_to_SDR(
                            seg, self.args.tester.distortion.fix_SDR.target_SDR
                        )

                    y = operator.degradation(seg)

                    xs.append(seg)
                    ys.append(y)
            operator = operator.to(self.device)

        if self.args.tester.set.mode == "stack_cut_end":
            # ys is a list of tensors, each tensor is an audio waveform
            # we stack them and cut the audio to the desired length
            # it could be that the audio is longer than the desired length at which the diffusion model works, so we cut it

            if self.args.tester.set.num_examples > 1:
                num_examples = min(len(ys), self.args.tester.set.num_examples)
            else:
                num_examples = len(ys)

            # initialize tensors
            y = torch.zeros(
                (num_examples, self.args.exp.audio_len), device=ys[0].device
            )
            x = torch.zeros(
                (num_examples, self.args.exp.audio_len), device=xs[0].device
            )

            for i in range(num_examples):

                print("ys[i].shape[0]", i, ys[i].shape[0], self.args.exp.audio_len)
                if ys[i].shape[0] > self.args.exp.audio_len:
                    print("cutting y" + str(i))

                y[i] = ys[i][0 : self.args.exp.audio_len]

                if xs[i].shape[0] > self.args.exp.audio_len:
                    print("cutting reference" + str(i))
                x[i] = xs[i][0 : self.args.exp.audio_len]

        else:
            raise NotImplementedError(
                f"mode {self.args.tester.set.mode} not implemented"
            )

        if is_test_set:
            x_test = []
            y_test = []

            if self.args.tester.distortion.target_type == "PairedDataset":

                for i, (original, distorted) in enumerate(
                    tqdm(self.inference_test_set)
                ):
                    seg = torch.from_numpy(original).float().to(self.device)
                    y = torch.from_numpy(distorted).float().to(self.device)

                    x_test.append(seg)
                    y_test.append(y)
            else:
                assert operator is not None, "Operator is None, should not be None"

                with torch.no_grad():
                    for i, data in enumerate(tqdm(self.inference_test_set)):
                        # check if the data is a tuple or a single element
                        if isinstance(data, tuple):
                            if len(data) == 2:
                                original, _ = data
                            else:
                                raise ValueError("The tuple should have 2 elements")
                        else:
                            original = data

                        seg = torch.from_numpy(original).float().to(self.device)

                        if self.args.tester.distortion.fix_SDR.use:
                            operator.adapt_params_to_SDR(
                                seg, self.args.tester.distortion.fix_SDR.target_SDR
                            )

                        y = operator.degradation(seg)

                        x_test.append(seg)
                        y_test.append(y)

            print("len(x_test)", len(x_test), "x_test[0].shape", x_test[0].shape)
        else:
            x_test = None
            y_test = None

        logging_callback = hydra.utils.instantiate(self.args.tester.logging_callback)
        evaluation_callback = hydra.utils.instantiate(
            self.args.tester.evaluation_callback
        )
        init_callback = hydra.utils.instantiate(self.args.tester.init_callback)

        path_dict = {
            "original": self.paths[mode + "original"],
            "degraded": self.paths[mode + "degraded"],
            "reconstructed": self.paths[mode + "reconstructed"],
        }
        if blind:
            path_dict["degraded_reconstructed"] = self.paths[
                mode + "degraded_reconstructed"
            ]
            path_dict["operator"] = self.paths[mode + "operator"]

        my_evaluation_callback = partial(
            evaluation_callback,
            x_test=x_test,
            y_test=y_test,
            args=self.args,
            paths=path_dict,
            blind=blind,
        )

        my_init_callback = partial(init_callback, args=self.args)
        my_logging_callback = logging_callback

        if blind:
            if self.args.tester.blind_distortion.op_type == "WienerHammerstein":
                from testing.operators.WienerHammerstein import WienerHammerstein

                operator_blind = WienerHammerstein(
                    self.args.tester.blind_distortion.op_hp, self.args.exp.sample_rate
                ).to(self.device)
            elif self.args.tester.blind_distortion.op_type == "Nonlinear":
                from testing.operators.NonlinearModule import Nonlinear

                operator_blind = Nonlinear(
                    self.args.tester.blind_distortion.op_hp,
                    self.args.exp.sample_rate,
                    y[0],
                ).to(self.device)
            elif self.args.tester.blind_distortion.op_type == "Wavenet":
                from testing.operators.operator_wavenet import WaveNetOperator

                operator_blind = WaveNetOperator(
                    self.args.tester.blind_distortion.op_hp_wavenet,
                    self.args.exp.sample_rate,
                )
            elif self.args.tester.blind_distortion.op_type == "Wavenet_GANmp":
                from testing.operators.operator_wavenet import WaveNetOperator

                operator_blind = WaveNetOperator(
                    self.args.tester.blind_distortion.op_hp_wavenet,
                    self.args.exp.sample_rate,
                    type="GANmp",
                )
            elif self.args.tester.blind_distortion.op_type == "S4":
                from testing.operators.s4_operator import S4Operator

                operator_blind = S4Operator(
                    self.args.tester.blind_distortion.op_hp_s4,
                    self.args.exp.sample_rate,
                )
            elif self.args.tester.blind_distortion.op_type == "GRU":
                from testing.operators.operator_GRU import GRUOperator

                operator_blind = GRUOperator(
                    self.args.tester.blind_distortion.op_hp_GRU,
                    self.args.exp.sample_rate,
                )
            elif self.args.tester.blind_distortion.op_type == "LSTM":
                from testing.operators.operator_LSTM import LSTMOperator

                operator_blind = LSTMOperator(
                    self.args.tester.blind_distortion.op_hp_GRU,
                    self.args.exp.sample_rate,
                )
        else:
            operator_blind = None

        print("Launching sampler predict conditional")
        pred = self.sampler.predict_conditional(
            y,
            operator_blind=operator_blind if blind else operator,
            reference=x,
            blind=blind,
            logging_callback=my_logging_callback,
            init_callback=my_init_callback,
            operator_ref=(
                operator
                if self.args.tester.distortion.target_type != "PairedDataset"
                else None
            ),
            evaluation_callback=my_evaluation_callback,
            save_path=self.paths[mode],
        )

    def test_independent(self, mode, blind=False):

        assert (
            self.args.tester.set.mode == "independent"
        ), "This function is only for independent mode"
        if self.inference_train_set is None:
            print("No inference train set specified")
            return
        if len(self.inference_train_set) == 0:
            print("No samples found in test set")
            return
        is_test_set = True
        if self.inference_test_set is None:
            print("No inference test set specified")
            print("not possible to run evaluation")
            is_test_set = False
        elif len(self.inference_test_set) == 0:
            print("No samples found in test set")
            is_test_set = False

        file_ids, degradeds, originals, preds = [], [], [], []

        print("Files will be saved in: ", self.paths[mode])

        # Setting of the known distortion operator
        # Load the previously genereated degraded samples - we dont know the original system at this point
        # We will degrade the signal with the known operator and then try to reconstruct it

        logging_callback = hydra.utils.instantiate(self.args.tester.logging_callback)
        evaluation_callback = hydra.utils.instantiate(
            self.args.tester.evaluation_callback
        )
        init_callback = hydra.utils.instantiate(self.args.tester.init_callback)

        my_init_callback = partial(init_callback, args=self.args)
        my_logging_callback = logging_callback

        for i, data in enumerate(tqdm(self.inference_train_set)):

            self.prepare_directories(
                mode, unconditional=False, blind=blind, string=str(i)
            )
            # redefine the logging paths according to the new paths (they have the string i in them)
            path_dict = {
                "original": self.paths[mode + "original"],
                "degraded": self.paths[mode + "degraded"],
                "reconstructed": self.paths[mode + "reconstructed"],
                "operator_ref": self.paths[mode + "operator_ref"],
            }
            if blind:
                path_dict["operator"] = self.paths[mode + "operator"]
            # if original file exists, we skip

            if (
                os.path.exists(os.path.join(self.paths[mode + "original"], "x.wav"))
                and not self.args.tester.set.overwrite
            ):
                print("skipping", i)
                continue

            if self.args.tester.distortion.target_type == "PairedDataset":
                original, distorted = data
                assert (
                    blind == True
                ), "informed distortion not supported for PairedDataset"
                seg = torch.from_numpy(original).float().to(self.device)
                y = torch.from_numpy(distorted).float().to(self.device)

                assert (
                    seg.shape == y.shape
                ), "original and distorted should have the same shape"

            else:
                if self.args.tester.distortion.target_type == "WienerHammerstein":
                    from testing.operators.WienerHammerstein import WienerHammerstein

                    operator = WienerHammerstein(
                        self.args.tester.distortion.op_hp, self.args.exp.sample_rate
                    ).to(self.device)
                elif self.args.tester.distortion.target_type == "LPF":
                    from testing.operators.lowpass_filters import LPF

                    operator = LPF(
                        self.args.tester.distortion.op_hp, self.args.exp.sample_rate
                    ).to(self.device)
                elif self.args.tester.distortion.target_type == "Filter":
                    from testing.operators.filters import Filter

                    operator = Filter(
                        self.args.tester.distortion.op_hp, self.args.exp.sample_rate
                    ).to(self.device)
                elif self.args.tester.distortion.target_type == "Nonlinear":
                    from testing.operators.NonlinearModule import Nonlinear

                    operator = Nonlinear(
                        self.args.tester.distortion.op_hp, self.args.exp.sample_rate
                    ).to(self.device)
                elif self.args.tester.distortion.target_type == "Linear":
                    from testing.operators.LinearModule import Linear

                    operator = Linear(
                        self.args.tester.distortion.op_hp, self.args.exp.sample_rate
                    ).to(self.device)
                elif self.args.tester.blind_distortion.op_type == "LTI":
                    from testing.operators.LTIModule import LTI

                    operator = LTI(
                        self.args.tester.distortion.op_hp, self.args.exp.sample_rate
                    )

                with torch.no_grad():
                    # check if the data is a tuple or a single element
                    if isinstance(data, tuple):
                        if len(data) == 2:
                            original, _ = data
                        else:
                            raise ValueError("The tuple should have 2 elements")
                    else:
                        original = data

                    seg = torch.from_numpy(original).float().to(self.device)
                    if self.args.tester.set.normalize_rms is not None:
                        seg = self.args.tester.set.normalize_rms * seg / seg.std()

                    if self.args.tester.distortion.fix_SDR.use:
                        print(
                            "adapting to SDR"
                            + str(self.args.tester.distortion.fix_SDR.target_SDR)
                        )
                        operator.adapt_params_to_SDR(
                            seg, self.args.tester.distortion.fix_SDR.target_SDR
                        )

                    y = operator.degradation(seg)

            if blind:
                if self.args.tester.blind_distortion.op_type == "WienerHammerstein":
                    from testing.operators.WienerHammerstein import WienerHammerstein

                    operator_blind = WienerHammerstein(
                        self.args.tester.blind_distortion.op_hp,
                        self.args.exp.sample_rate,
                    ).to(self.device)
                elif self.args.tester.blind_distortion.op_type == "Linear":
                    from testing.operators.LinearModule import Linear

                    operator_blind = Linear(
                        self.args.tester.blind_distortion.op_hp,
                        self.args.exp.sample_rate,
                    ).to(self.device)
                elif self.args.tester.blind_distortion.op_type == "Nonlinear":
                    from testing.operators.NonlinearModule import Nonlinear

                    operator_blind = Nonlinear(
                        self.args.tester.blind_distortion.op_hp,
                        self.args.exp.sample_rate,
                        y[0],
                    ).to(self.device)
                elif self.args.tester.blind_distortion.op_type == "Wavenet":
                    from testing.operators.operator_wavenet import WaveNetOperator

                    operator_blind = WaveNetOperator(
                        self.args.tester.blind_distortion.op_hp_wavenet,
                        self.args.exp.sample_rate,
                    )
                elif self.args.tester.blind_distortion.op_type == "Wavenet_GANmp":
                    from testing.operators.operator_wavenet import WaveNetOperator

                    operator_blind = WaveNetOperator(
                        self.args.tester.blind_distortion.op_hp_wavenet,
                        self.args.exp.sample_rate,
                        type="GANmp",
                    )
                elif self.args.tester.blind_distortion.op_type == "S4":
                    from testing.operators.s4_operator import S4Operator

                    operator_blind = S4Operator(
                        self.args.tester.blind_distortion.op_hp_s4,
                        self.args.exp.sample_rate,
                    )
                elif self.args.tester.blind_distortion.op_type == "GRU":
                    from testing.operators.operator_GRU import GRUOperator

                    operator_blind = GRUOperator(
                        self.args.tester.blind_distortion.op_hp_GRU,
                        self.args.exp.sample_rate,
                    )
                elif self.args.tester.blind_distortion.op_type == "LSTM":
                    from testing.operators.operator_LSTM import LSTMOperator

                    operator_blind = LSTMOperator(
                        self.args.tester.blind_distortion.op_hp_GRU,
                        self.args.exp.sample_rate,
                    )
                elif self.args.tester.blind_distortion.op_type == "LTI":
                    from testing.operators.LTIModule import LTI

                    operator_blind = LTI(
                        self.args.tester.blind_distortion.op_hp,
                        self.args.exp.sample_rate,
                    )

                if self.args.tester.blind_distortion.op_hp.init_params.use == True:
                    with torch.no_grad():
                        # check if the data is a tuple or a single element
                        if isinstance(data, tuple):
                            if len(data) == 2:
                                original, _ = data
                            else:
                                raise ValueError("The tuple should have 2 elements")
                        else:
                            original = data

                        seg = torch.from_numpy(original).float().to(self.device)
                        if self.args.tester.set.normalize_rms is not None:
                            seg = self.args.tester.set.normalize_rms * seg / seg.std()

                        print(f"Initializing the spline coefs")
                        operator_blind.initialize_coeffs(y)
            else:
                operator_blind = None

            x = seg.view(1, -1)
            y = y.view(1, -1)
            my_evaluation_callback = partial(
                evaluation_callback,
                x_test=None,
                y_test=None,
                args=self.args,
                paths=path_dict,
                blind=blind,
            )

            pred = self.sampler.predict_conditional(
                y,
                operator_blind=operator_blind if blind else operator,
                reference=x,
                blind=blind,
                logging_callback=my_logging_callback,
                init_callback=my_init_callback,
                operator_ref=(
                    operator
                    if self.args.tester.distortion.target_type != "PairedDataset"
                    else None
                ),
                evaluation_callback=my_evaluation_callback,
                save_path=self.paths[mode],
            )

    def prepare_directories(self, mode, unconditional=False, blind=False, string=None):

        today = date.today()
        self.paths = {}
        if (
            "overriden_name" in self.args.tester.keys()
            and self.args.tester.overriden_name is not None
        ):
            self.path_sampling = os.path.join(
                self.args.model_dir, self.args.tester.overriden_name
            )
        else:
            self.path_sampling = os.path.join(
                self.args.model_dir, "test" + today.strftime("%d_%m_%Y")
            )
        if not os.path.exists(self.path_sampling):
            os.makedirs(self.path_sampling)

        self.paths[mode] = os.path.join(
            self.path_sampling, mode, self.args.exp.exp_name
        )

        if not os.path.exists(self.paths[mode]):
            os.makedirs(self.paths[mode])
        if string is None:
            string = ""

        if not unconditional:
            self.paths[mode + "original"] = os.path.join(
                self.paths[mode], string + "original"
            )
            if not os.path.exists(self.paths[mode + "original"]):
                os.makedirs(self.paths[mode + "original"])
            self.paths[mode + "degraded"] = os.path.join(
                self.paths[mode], string + "degraded"
            )
            if not os.path.exists(self.paths[mode + "degraded"]):
                os.makedirs(self.paths[mode + "degraded"])
            self.paths[mode + "reconstructed"] = os.path.join(
                self.paths[mode], string + "reconstructed"
            )
            if not os.path.exists(self.paths[mode + "reconstructed"]):
                os.makedirs(self.paths[mode + "reconstructed"])
            self.paths[mode + "operator_ref"] = os.path.join(
                self.paths[mode], string + "operator_ref"
            )
            if not os.path.exists(self.paths[mode + "operator_ref"]):
                os.makedirs(self.paths[mode + "operator_ref"])
            if blind:
                self.paths[mode + "operator"] = os.path.join(
                    self.paths[mode], string + "operator"
                )
                if not os.path.exists(self.paths[mode + "operator"]):
                    os.makedirs(self.paths[mode + "operator"])

    def save_experiment_args(self, mode):
        with open(
            os.path.join(self.paths[mode], ".argv"), "w"
        ) as f:  # Keep track of the arguments we used for this experiment
            omegaconf.OmegaConf.save(config=self.args, f=f.name)

    def do_test(self, it=0):

        self.it = it
        print(self.args.tester.modes)
        for m in self.args.tester.modes:

            if m == "unconditional":
                print("testing unconditional")
                if not self.in_training:
                    self.prepare_directories(m, unconditional=True)
                    self.save_experiment_args(m)
                self.sample_unconditional(m)
            elif m == "blind":
                assert self.inference_train_set is not None
                print("testing blind distortion ")
                if not self.in_training:
                    self.prepare_directories(m, unconditional=False, blind=True)
                    self.save_experiment_args(m)

                if self.args.tester.set.mode == "independent":
                    self.test_independent(m, blind=True)
                else:
                    self.test(m, blind=True)
            elif m == "informed":
                assert self.inference_train_set is not None
                print("testing informed distortion")
                if not self.in_training:
                    self.prepare_directories(m, unconditional=False)
                    self.save_experiment_args(m)
                if self.args.tester.set.mode == "independent":
                    self.test_independent(m, blind=False)
                else:
                    self.test(m, blind=False)
            elif m == "oracle":
                assert self.inference_train_set is not None
                print("testing oracle distortion ")
                if not self.in_training:
                    self.prepare_directories(m, unconditional=False, blind=True)
                    self.save_experiment_args(m)

                if self.args.tester.set.mode == "independent":
                    self.test_independent(m, blind=True)
                else:
                    self.test(m, blind=True)
            elif m == "conditional_supervised":
                assert self.inference_test_set is not None

                self.test_conditional_supervised(m)
            elif m == "conditional_supervised_evaluation":
                assert self.inference_test_set is not None
                self.prepare_directories(m, unconditional=False, blind=True)
                self.test_conditional_supervised_evaluation(m)
            else:
                print("Warning: unknown mode: ", m)
