import os
import torch
from datetime import datetime


def build_model_name(args, config):
    """
    Creates a descriptive model name from arguments.
    """

    parts = [
        args.lang,
        f"task{args.task}",
        args.mode
    ]

    if args.peft:
        parts.append(args.peft)

    if args.k:
        parts.append(f"k{args.k}")

    parts.append(f"lr{config['learning_rate']}")
    parts.append(f"ep{config['epochs']}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    return "_".join(parts) + "_" + timestamp + ".pt"


def save_model(model, args, config, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)

    filename = build_model_name(args, config)
    path = os.path.join(save_dir, filename)

    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "config": config
    }, path)

    print(f"\nModel saved to: {path}")