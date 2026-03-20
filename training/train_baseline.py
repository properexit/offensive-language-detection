from models.baseline import BaselineModel
from datasets.english.loaders_olid import load_task_a_olid

def main():
    # load English Task A split
    train_df, dev_df = load_task_a_olid()

    # DEBUG: reduce dataset size
    # DEBUG_N = 200
    # train_df = train_df.sample(n=min(DEBUG_N, len(train_df)), random_state=42)
    # dev_df = dev_df.sample(n=min(DEBUG_N, len(dev_df)), random_state=42)

    baseline = BaselineModel()

    # run both majority and TF-IDF baselines
    baseline.run_all(
        train_df["tweet"],
        train_df["label"],
        dev_df["tweet"],
        dev_df["label"],
        target_names=["NOT", "OFF"]
    )


if __name__ == "__main__":
    main()