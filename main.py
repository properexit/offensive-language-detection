import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

from transformers import logging
logging.set_verbosity_error()

import argparse
import yaml

from utils.device import get_device
from training.train_transformer import train_transformer
from training.train_multitask import train_multitask  

from datasets.english.loaders_olid import (   
    load_task_a_olid,
    load_task_b_olid,
    load_task_c_olid
)

from datasets.arabic.loaders import load_task_a_arabic

from utils.seed import set_seed

MODEL_MAP = {
    "english": "google/bert_uncased_L-2_H-128_A-2",
    "arabic": "xlm-roberta-base"
}


def main():
    parser = argparse.ArgumentParser(
        description="Offensive language detection (English + Arabic)"
    )

    parser.add_argument("--lang", choices=["english", "arabic"])
    parser.add_argument("--task", choices=["A", "B", "C"])

    parser.add_argument(
        "--mode",
        default="finetune",
        choices=["finetune", "zero-shot", "few-shot"]
    )

    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--device", default="auto")

    # PEFT 
    parser.add_argument(
        "--peft",
        choices=["lora", "freeze"],
        default=None
    )

    # Multitask flag
    parser.add_argument(
        "--multitask",
        action="store_true",
        help="Run English multi-task (A + B)"
    )

    args = parser.parse_args()

    # multitask shortcut
    if args.multitask:
        print("Running multi-task training (English A + B)")
        train_multitask()
        return

    if args.lang is None or args.task is None:
        parser.error("--lang and --task are required unless --multitask is used.")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))
    
    device = get_device(
        prefer_gpu=(args.device == "auto"),
        force_device=None if args.device == "auto" else args.device
    )

    print("Device:", device)
    print("Lang:", args.lang,
          "| Task:", args.task,
          "| Mode:", args.mode,
          "| PEFT:", args.peft)   

    # English
    if args.lang == "english":

        loader_map = {
            "A": load_task_a_olid,
            "B": load_task_b_olid,
            "C": load_task_c_olid,
        }

        train_df, dev_df = loader_map[args.task]()

        train_df, dev_df = loader_map[args.task]()

        # DEBUG: reduce dataset size for quick testing
        # DEBUG_N = 200
        # train_df = train_df.sample(n=min(DEBUG_N, len(train_df)), random_state=42)
        # dev_df = dev_df.sample(n=min(DEBUG_N, len(dev_df)), random_state=42)

        num_labels = 2 if args.task != "C" else 3

        train_transformer(
            model_name=MODEL_MAP["english"],
            train_df=train_df,
            dev_df=dev_df,
            num_labels=num_labels,
            language="english",
            mode=args.mode,
            few_shot_k=args.k,
            config=config,
            device=device,
            peft_type=args.peft   
        )
        return

    # Arabic (Task A only)
    if args.task != "A":
        raise ValueError("Arabic only supports Task A")

    if args.mode == "zero-shot":
        train_df, dev_df = load_task_a_arabic()

        train_transformer(
            model_name=MODEL_MAP["arabic"],
            train_df=train_df,
            dev_df=dev_df,
            num_labels=2,
            language="arabic",
            mode="zero-shot",
            few_shot_k=None,
            config=config,
            device=device,
            peft_type=args.peft   
        )
        return

    # few-shot with English pretraining
    print("Pretraining on English first...")

    en_train, en_dev = load_task_a_olid()

    # DEBUG: reduce pretraining size
    # DEBUG_N = 200
    # en_train = en_train.sample(n=min(DEBUG_N, len(en_train)), random_state=42)
    # en_dev = en_dev.sample(n=min(DEBUG_N, len(en_dev)), random_state=42)

    model = train_transformer(
        model_name=MODEL_MAP["arabic"],
        train_df=en_train,
        dev_df=en_dev,
        num_labels=2,
        language="english",
        mode="finetune",
        few_shot_k=None,
        config=config,
        device=device,
        return_model=True,
        peft_type=args.peft   
    )

    print("Fine-tuning on Arabic...")

    ar_train, ar_dev = load_task_a_arabic()

    # DEBUG: reduce pretraining size
    # ar_train = ar_train.sample(n=min(DEBUG_N, len(ar_train)), random_state=42)
    # ar_dev = ar_dev.sample(n=min(DEBUG_N, len(ar_dev)), random_state=42)

    train_transformer(
        model_name=MODEL_MAP["arabic"],
        train_df=ar_train,
        dev_df=ar_dev,
        num_labels=2,
        language="arabic",
        mode="few-shot",
        few_shot_k=args.k,
        config=config,
        device=device,
        model=model,
        peft_type=args.peft   
    )


if __name__ == "__main__":
    main()