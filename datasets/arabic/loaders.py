import pandas as pd
from sklearn.model_selection import train_test_split


def load_task_a_arabic(split_ratio=0.2):
    """
    Load Arabic OffensEval Task A training data.
    """

    tweets = []
    labels = []

    path = "data/raw/arabic/offenseval-ar-training-v1/offenseval-ar-training-v1.tsv"

    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")

            if len(parts) < 3:
                continue

            tweet = parts[1]
            label = parts[2]

            tweets.append(tweet.replace("<LF>", " "))
            labels.append(1 if label == "OFF" else 0)

    df = pd.DataFrame({
        "tweet": tweets,
        "label": labels
    })

    train_df, dev_df = train_test_split(
        df,
        test_size=split_ratio,
        stratify=df["label"],
        random_state=42
    )

    return train_df, dev_df