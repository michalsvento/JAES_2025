import wandb


def init_callback(wandb_run, dict, args, device, step):
    wandb_run.log(
        {
            "y": [
                wandb.Audio(
                    dict["y"][0:8].view(-1).detach().cpu(),
                    caption="y",
                    sample_rate=args.exp.sample_rate,
                )
            ]
        },
        step=step,
    )
    wandb_run.log(
        {
            "reference": [
                wandb.Audio(
                    dict["reference"][0:8].view(-1).detach().cpu(),
                    caption="reference",
                    sample_rate=args.exp.sample_rate,
                )
            ]
        },
        step=step,
    )
