import pandas as pd
from sklearn.model_selection import train_test_split


def _split(df, split_ratio=0.2, seed=42):
    return train_test_split(
        df,
        test_size=split_ratio,
        stratify=df["label"],
        random_state=seed
    )


def load_task_a_english(split_ratio=0.2):
    tweets = pd.read_csv("data/raw/english/test_a_tweets.tsv", sep="\t")
    labels = pd.read_csv("data/raw/english/test_a_labels.csv", header=None)
    labels.columns = ["id", "label"]

    labels["label"] = labels["label"].map({"NOT": 0, "OFF": 1})
    df = tweets.merge(labels, on="id")

    return _split(df, split_ratio)


def load_task_b_english(split_ratio=0.2):
    tweets = pd.read_csv("data/raw/english/test_b_tweets.tsv", sep="\t")
    labels = pd.read_csv("data/raw/english/test_b_labels.csv", header=None)
    labels.columns = ["id", "label"]

    labels["label"] = labels["label"].map({"UNT": 0, "TIN": 1})
    df = tweets.merge(labels, on="id")

    return _split(df, split_ratio)


def load_task_c_english(split_ratio=0.2):
    tweets = pd.read_csv("data/raw/english/test_c_tweets.tsv", sep="\t")
    labels = pd.read_csv("data/raw/english/test_c_labels.csv", header=None)
    labels.columns = ["id", "label"]

    labels["label"] = labels["label"].map({
        "IND": 0,
        "GRP": 1,
        "OTH": 2
    })

    df = tweets.merge(labels, on="id")

    return _split(df, split_ratio)