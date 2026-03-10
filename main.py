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

from datasets.english.loaders_olid import (
    load_task_a_olid,
    load_task_b_olid,
    load_task_c_olid
)

from datasets.arabic.loaders import load_task_a_arabic

from utils.seed import set_seed
set_seed(42)

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

    args = parser.parse_args()

    if args.lang is None or args.task is None:
        parser.error("--lang and --task are required.")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device(
        prefer_gpu=(args.device == "auto"),
        force_device=None if args.device == "auto" else args.device
    )

    print("Device:", device)
    print("Lang:", args.lang,
          "| Task:", args.task,
          "| Mode:", args.mode)

    # English
    if args.lang == "english":

        loader_map = {
            "A": load_task_a_olid,
            "B": load_task_b_olid,
            "C": load_task_c_olid,
        }

        train_df, dev_df = loader_map[args.task]()
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
            device=device
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
            device=device
        )
        return

    # few-shot with English pretraining
    print("Pretraining on English first...")

    en_train, en_dev = load_task_a_olid()

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
        return_model=True
    )

    print("Fine-tuning on Arabic...")

    ar_train, ar_dev = load_task_a_arabic()

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
        model=model
    )
if __name__ == "__main__":
    main()