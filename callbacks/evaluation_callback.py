import numpy as np
import torch
import os
import wandb
import plotly.express as px
import plotly.graph_objects as go
import utils.testing_utils as tt_utils
import utils.log as utils_logging
import soundfile as sf
import pickle


# import dill


def evaluation_callback_notest(
    step, dict, metrics, x_test, y_test, wandb_run, device, args, paths, blind=True
):

    x_train = dict["reference"]
    x_den = dict["x_den"]
    operator = dict["operator"]
    operator_ref = dict["operator_ref"]

    shape = x_train.shape
    if x_den is not None:
        x_den = x_den.view(-1, shape[-1])
        y_hats_tweedie = operator.degradation(x_den)

    y_train = dict["y"]

    # train set
    reference_train = dict["reference"]
    y_hats_train = operator.degradation(reference_train)

    # save the data to paths

    if step == 0:
        sf.write(
            os.path.join(paths["degraded"], "y.wav"),
            y_train[0].detach().cpu().numpy(),
            args.exp.sample_rate,
        )
        sf.write(
            os.path.join(paths["original"], "x.wav"),
            reference_train[0].detach().cpu().numpy(),
            args.exp.sample_rate,
        )
        if wandb_run is not None:
            fig = utils_logging.plot_STFT(
                y_train.view(-1).detach().cpu(), operator, type="mag_dB"
            )
            wandb_run.log({"y_spec": fig}, step=step)

        if operator_ref is not None:
            operator_cpu_ref = operator_ref.to("cpu")
            file_operator = torch.save(
                operator_cpu_ref.state_dict(),
                open(
                    os.path.join(
                        paths["operator_ref"], "operator_ref" + str(step) + ".pt"
                    ),
                    "wb",
                ),
            )

    if x_den is not None:
        sf.write(
            os.path.join(paths["reconstructed"], "x_hat" + str(step) + ".wav"),
            x_den[0].detach().cpu().numpy(),
            args.exp.sample_rate,
        )

    # save operator in a pickle file
    if blind:
        operator_cpu = operator.to("cpu")
        file_operator = torch.save(
            operator_cpu.state_dict(),
            open(os.path.join(paths["operator"], "operator" + str(step) + ".pt"), "wb"),
        )

    try:
        for m in metrics:
            if m == "ESR":
                ESR = tt_utils.compute_ESR(y_train, y_hats_train)
                print(f"ESR_FWR_train: {ESR}")
                wandb_run.log({f"ESR_FWR_train": ESR}, step=step)

                ESR = tt_utils.compute_ESR(y_train, y_hats_tweedie)
                print(f"ESR_FWR_train_tweedie: {ESR}")
                wandb_run.log({f"ESR_FWR_train_tweedie": ESR}, step=step)

                ESR = tt_utils.compute_ESR(x_train, x_den)
                print(f"ESR_FWR_clean_train: {ESR}")
                wandb_run.log({f"ESR_FWR_clea_train": ESR}, step=step)

            if m == "STFT_mag":
                STFT_mag = tt_utils.compute_STFT_mag(
                    y_train, y_hats_train, fft_size=1024, hop_size=256
                )
                print(f"STFT_mag_train: {STFT_mag}")
                wandb_run.log({f"STFT_mag_train": STFT_mag}, step=step)

                STFT_mag = tt_utils.compute_STFT_mag(
                    y_train, y_hats_tweedie, fft_size=1024, hop_size=256
                )
                print(f"STFT_mag_train_tweedie: {STFT_mag}")
                wandb_run.log({f"STFT_mag_train_tweedie": STFT_mag}, step=step)

                STFT_mag = tt_utils.compute_STFT_mag(
                    x_train, x_den, fft_size=1024, hop_size=256
                )
                print(f"STFT_mag__clean_train: {STFT_mag}")
                wandb_run.log({f"STFT_mag_clean_train": STFT_mag}, step=step)

            if m == "LSD":
                STFT_mag = tt_utils.compute_LSD(
                    y_train, y_hats_train, fft_size=1024, hop_size=256
                )
                print(f"LSD_train: {STFT_mag}")
                wandb_run.log({f"LSD_train": STFT_mag}, step=step)

                STFT_mag = tt_utils.compute_LSD(
                    y_train, y_hats_tweedie, fft_size=1024, hop_size=256
                )
                print(f"LSD_train_tweedie: {STFT_mag}")
                wandb_run.log({f"LSD_train_tweedie": STFT_mag}, step=step)

                STFT_mag = tt_utils.compute_LSD(
                    x_train, x_den, fft_size=1024, hop_size=256
                )
                print(f"LSD_clean_train: {STFT_mag}")
                wandb_run.log({f"LSD_clean_train": STFT_mag}, step=step)

            if m == "freq_error":
                freq_error = tt_utils.compute_freq_error(
                    y_train, y_hats_train, fft_size=1024, hop_size=256
                )
                fig = px.line(log_x=True)
                freqs = torch.linspace(0, args.exp.sample_rate / 2, 1024 // 2 + 1)
                fig.add_scatter(x=freqs, y=freq_error.detach().cpu())
                wandb_run.log({"Freq_error_train": fig}, step=step)

                freq_error = tt_utils.compute_freq_error(
                    y_train, y_hats_tweedie, fft_size=1024, hop_size=256
                )
                fig = px.line(log_x=True)
                freqs = torch.linspace(0, args.exp.sample_rate / 2, 1024 // 2 + 1)
                fig.add_scatter(x=freqs, y=freq_error.detach().cpu())
                wandb_run.log({"Freq_error_train_tweedie": fig}, step=step)

                freq_error = tt_utils.compute_freq_error(
                    x_train, x_den, fft_size=1024, hop_size=256
                )
                fig = px.line(log_x=True)
                freqs = torch.linspace(0, args.exp.sample_rate / 2, 1024 // 2 + 1)
                fig.add_scatter(x=freqs, y=freq_error.detach().cpu())
                wandb_run.log({"Freq_error_clean_train": fig}, step=step)

    except:
        pass

    try:
        impulse = torch.zeros(8000).to(device)
        impulse[32] = 1
        with torch.no_grad():
            impulse_response = operator.degradation(impulse)

        fig = px.line()
        fig.add_scatter(x=torch.arange(0, 8000).cpu(), y=impulse.cpu())
        fig.add_scatter(
            x=torch.arange(0, 8000).cpu(), y=impulse_response[0].detach().cpu()
        )
        fig.update_yaxes(range=[-1, 1])
    except:
        print("impulse_response not computed")
    else:
        wandb_run.log({"impulse_response": fig}, step=step)

    if True:
        ramp = torch.linspace(-1, 1, 1000).to(device)
        refer = dict["reference"]
        ref_min, ref_max = torch.min(refer), torch.max(refer)
        fig = px.line()
        fig.add_vrect(
            x0=ref_min.item(),
            x1=ref_max.item(),
            annotation_text="reference sample space",
            annotation_position="top left",
            fillcolor="orange",
            opacity=0.1,
            line_width=0,
        )
        with torch.no_grad():
            if operator_ref is not None:
                response = operator_ref.degradation(ramp)
            responsepred = operator.nld(ramp.view(1, -1)).squeeze(0)

        if operator_ref is not None:
            fig.add_scatter(
                x=ramp.view(-1).cpu().numpy(),
                y=response.view(-1).detach().cpu().numpy(),
                name="reference response",
            )
        fig.add_scatter(
            x=ramp.view(-1).cpu().numpy(),
            y=responsepred.view(-1).detach().cpu().numpy(),
            name="predicted response",
        )
        if operator.op_hp.NLD == "CCR":
            coefs = operator.params["spline_coefs"].view(-1).detach().cpu().numpy()
            grid = operator.grid.view(-1).detach().cpu().numpy()
            fig.add_trace(
                go.Scatter(x=grid, y=coefs, mode="markers", name="spline_coefs")
            )
        fig.update_yaxes(range=[-1, 1])
        wandb_run.log({"memoryless_transfer_function": fig}, step=step)

    if args.tester.blind_distortion.op_type in ["LTI", "WienerHammerstein"]:

        try:
            print("plotting LTI response")
            print("plotting LTI response - BLind")
            # LTI - magnitude and phase response
            with torch.no_grad():
                H_mag, H_phase = operator.get_LTI_response()
                frequencies = operator.freqs.view(-1).detach().cpu().numpy()
            print(H_mag.shape, H_phase.shape, H_mag.shape[0], frequencies.shape)
            for j in range(H_mag.shape[0]):
                fig = utils_logging.create_mag_phase_plot()
                fig.add_trace(
                    go.Scatter(
                        x=frequencies,
                        y=H_mag[j].view(-1).detach().cpu().numpy(),
                        mode="lines",
                    ),
                    secondary_y=False,
                )
                fig.add_trace(
                    go.Scatter(
                        x=frequencies,
                        y=H_phase[j].view(-1).detach().cpu().numpy(),
                        mode="lines",
                    ),
                    secondary_y=True,
                )
                fig.update_yaxes(range=[-100, 20])
                print(
                    f"Response LTI_{j + 1}, magnitude and phase, shape: {H_mag[j].view(-1).shape}, {H_phase[j].view(-1).shape}, freqs: {frequencies.shape}"
                )
        except Exception as e:
            print(e)
            print("LTI response not computed")
        else:
            for j in range(H_mag.shape[0]):
                wandb_run.log({f"Response LTI_{j + 1}": fig}, step=step)

    if args.tester.blind_distortion.op_type in ["LTI", "WienerHammerstein"]:

        try:
            print("plotting LTI response")
            print("plotting LTI response - Ref")
            # LTI - magnitude and phase response
            with torch.no_grad():
                H_mag, H_phase = operator_ref.get_LTI_response()
                frequencies = operator.freqs.view(-1).detach().cpu().numpy()
            print(H_mag.shape, H_phase.shape, H_mag.shape[0], frequencies.shape)
            for j in range(H_mag.shape[0]):
                fig = utils_logging.create_mag_phase_plot()
                fig.add_trace(
                    go.Scatter(
                        x=frequencies,
                        y=H_mag[j].view(-1).detach().cpu().numpy(),
                        mode="lines",
                    ),
                    secondary_y=False,
                )
                fig.add_trace(
                    go.Scatter(
                        x=frequencies,
                        y=H_phase[j].view(-1).detach().cpu().numpy(),
                        mode="lines",
                    ),
                    secondary_y=True,
                )
                fig.update_yaxes(range=[-100, 20])
                print(
                    f"Response Reference LTI_{j + 1}, magnitude and phase, shape: {H_mag[j].view(-1).shape}, {H_phase[j].view(-1).shape}, freqs: {frequencies.shape}"
                )
        except Exception as e:
            print(e)
            print("LTI response not computed")
        else:
            for j in range(H_mag.shape[0]):
                wandb_run.log({f"Response reference LTI_{j + 1}": fig}, step=step)

    try:
        fig = go.Figure()
        coefs = operator.params["spline_coefs"].view(-1).detach().cpu().numpy()
        grid = operator.grid.view(-1).detach().cpu().numpy()
        alphas = operator.params["alphas"].view(-1).detach().cpu().numpy()
        print(coefs.shape, grid.shape, alphas.shape)
        fig.add_trace(go.Scatter(x=grid, y=coefs, mode="lines", name="spline_coefs"))
        fig.add_trace(go.Scatter(x=grid[1:-2], y=alphas, mode="lines", name="alphas"))
        wandb_run.log({f"Learnable params": fig}, step=step)
    except Exception as e:
        print(e)
        print("spline_coefs not computed")

    try:
        fig = go.Figure()

        bins_settings = {"start": -1.025, "end": 1.025, "size": 1 / 20}

        maxim = torch.max(torch.abs(dict["x"])).detach().cpu().numpy()
        if maxim < 1:
            maxim = 1
        wandb_run.log(
            {
                "x": wandb.Audio(
                    dict["x"][0:8].view(-1).detach().cpu().numpy() / maxim,
                    sample_rate=args.exp.sample_rate,
                )
            },
            step=step,
        )
        fig.add_trace(
            go.Histogram(
                x=dict["x"][0:8].view(-1).detach().cpu().numpy(),
                name="x",
                opacity=0.6,
                autobinx=False,
                xbins=bins_settings,
            )
        )

        maxim = torch.max(torch.abs(x_den)).detach().cpu().numpy()
        if maxim < 1:
            maxim = 1

        wandb_run.log(
            {
                "x_tweedie": wandb.Audio(
                    x_den[0:8].view(-1).detach().cpu() / maxim,
                    sample_rate=args.exp.sample_rate,
                )
            },
            step=step,
        )
        fig.add_trace(
            go.Histogram(
                x=x_den[0:8].view(-1).detach().cpu().numpy(),
                name="x_tweedie",
                opacity=0.6,
                autobinx=False,
                xbins=bins_settings,
            )
        )

        wandb_run.log(
            {
                "x_tweedie_spec": utils_logging.plot_STFT(
                    dict["x_den"][0:8].view(-1).detach(), operator
                )
            },
            step=step,
        )

        with torch.no_grad():
            y_recon = operator.degradation(x_den)

        maxim = 1
        wandb_run.log(
            {
                "y_rec_tweedie": wandb.Audio(
                    y_recon[0:8].view(-1).detach().cpu().numpy() / maxim,
                    sample_rate=args.exp.sample_rate,
                )
            },
            step=step,
        )
        fig.add_trace(
            go.Histogram(
                x=y_recon[0:8].view(-1).detach().cpu().numpy(),
                name="y_rec_tweedie",
                opacity=0.6,
                autobinx=False,
                xbins=bins_settings,
            )
        )

        with torch.no_grad():
            reference = dict["reference"]
            y_recon = operator.degradation(reference)

        maxim = 1
        wandb_run.log(
            {
                "y_rec_ref": wandb.Audio(
                    y_recon[0:8].view(-1).detach().cpu().numpy() / maxim,
                    sample_rate=args.exp.sample_rate,
                )
            },
            step=step,
        )
        fig.add_trace(
            go.Histogram(
                x=y_recon[0:8].view(-1).detach().cpu().numpy(),
                name="y_rec_ref",
                opacity=0.6,
                autobinx=False,
                xbins=bins_settings,
            )
        )

        fig.update_layout(
            barmode="overlay", title_text="Overlaid Histograms", height=600, width=800
        )
    except:
        pass

    # Histogram plots
    try:
        wandb_run.log({"Histogram": fig}, step=step)
    except Exception as e:
        print(e)

    try:
        grads = dict["grads"]
        grads_spec = utils_logging.plot_STFT(grads[0:8].view(-1).detach(), operator)
        wandb_run.log({"lhscore_spec": grads_spec}, step=step)

    except Exception as e:
        print("grads not logged", e)
        print(e)

    try:
        score_spec = utils_logging.plot_STFT(
            dict["score"][0:8].view(-1).detach(), operator
        )
        wandb_run.log({"score_spec": score_spec}, step=step)
    except Exception as e:
        print("score not logged", e)
        print(e)
