import torch
import wandb


def logging_callback(wandb_run, dict, step_counter):
    print("step_counter", step_counter)

    wandb_run.log({"rec_loss": dict["rec_loss_value"]}, step=step_counter)
    try:
        wandb_run.log({"rec_loss_aux": dict["rec_loss_aux_value"]}, step=step_counter)
    except:
        pass
    wandb_run.log({"lh_score": dict["lh_score_norm"]}, step=step_counter)
    wandb_run.log({"score": dict["score_norm"]}, step=step_counter)
    wandb_run.log({"t_i": dict["t_i"]}, step=step_counter)
    wandb_run.log({"t_iplus1": dict["t_iplus1"]}, step=step_counter)


def logging_callback_PF(wandb_run, dict, step_counter):
    print("step_counter", step_counter)

    wandb_run.log(
        {
            "rec_losses": wandb.Histogram(
                dict["rec_loss_values"].detach().cpu(), num_bins=100
            )
        },
        step=step_counter,
    )
    try:
        wandb_run.log({"rec_loss_aux": dict["rec_loss_aux_value"]}, step=step_counter)
    except:
        pass
    wandb_run.log({"lh_score": dict["lh_score_norm"]}, step=step_counter)
    wandb_run.log({"score": dict["score_norm"]}, step=step_counter)
    wandb_run.log({"t_i": dict["t_i"]}, step=step_counter)
    wandb_run.log({"t_iplus1": dict["t_iplus1"]}, step=step_counter)
    wandb_run.log(
        {
            "normalized_weights": wandb.Histogram(
                dict["normalized_weights"].detach().cpu()
            )
        },
        step=step_counter,
    )
