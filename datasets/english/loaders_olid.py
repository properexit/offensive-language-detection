import pandas as pd
from sklearn.model_selection import train_test_split


def _split(df, split_ratio=0.2, seed=42):
    return train_test_split(
        df,
        test_size=split_ratio,
        stratify=df["label"],
        random_state=seed
    )


def _load_olid():
    df = pd.read_csv(
        "data/raw/english_OLID/olid-training-v1.0.tsv",
        sep="\t"
    )
    return df

# Task A
def load_task_a_olid(split_ratio=0.2):

    df = _load_olid()

    df = df[["tweet", "subtask_a"]]

    df["label"] = df["subtask_a"].map({
        "NOT": 0,
        "OFF": 1
    })

    df = df[["tweet", "label"]]

    return _split(df, split_ratio)

# Task B
def load_task_b_olid(split_ratio=0.2):

    df = _load_olid()

    df = df[df["subtask_b"] != "NULL"]

    df = df[["tweet", "subtask_b"]]

    df["label"] = df["subtask_b"].map({
        "UNT": 0,
        "TIN": 1
    })

    df = df[["tweet", "label"]]

    return _split(df, split_ratio)

# Task C
def load_task_c_olid(split_ratio=0.2):

    df = _load_olid()

    df = df[df["subtask_c"] != "NULL"]

    df = df[["tweet", "subtask_c"]]

    df["label"] = df["subtask_c"].map({
        "IND": 0,
        "GRP": 1,
        "OTH": 2
    })

    df = df[["tweet", "label"]]

    return _split(df, split_ratio)