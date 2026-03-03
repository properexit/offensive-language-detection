from models.baseline import BaselineModel
from datasets.english.loaders import load_task_a_english


def main():
    # load English Task A split
    train_df, dev_df = load_task_a_english()

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