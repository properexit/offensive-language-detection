from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report


class BaselineModel:
    """
    Simple baselines for offensive language detection:
    1) Majority class predictor
    2) TF-IDF + Logistic Regression
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20000
        )
        self.clf = LogisticRegression(max_iter=1000, random_state=42)

    def majority_baseline(self, train_labels, dev_labels, target_names=None):
        # predict the most frequent label from training set
        majority_label = Counter(train_labels).most_common(1)[0][0]
        preds = [majority_label] * len(dev_labels)

        print("\nMajority baseline")
        print("Predicted label:", majority_label)
        print("Macro F1:",
              f1_score(dev_labels, preds, average="macro"))
        print(classification_report(
            dev_labels,
            preds,
            target_names=target_names,
            zero_division=0
        ))

    def fit(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)

    def evaluate(self, texts, labels, target_names=None):
        X = self.vectorizer.transform(texts)
        preds = self.clf.predict(X)

        print("\nTF-IDF + Logistic Regression")
        print("Macro F1:",
              f1_score(labels, preds, average="macro"))
        print(classification_report(
            labels,
            preds,
            target_names=target_names,
            zero_division=0
        ))

    def run_all(self, train_texts, train_labels,
                dev_texts, dev_labels,
                target_names=None):

        # majority baseline 
        self.majority_baseline(train_labels, dev_labels, target_names)

        # classical ML baseline
        self.fit(train_texts, train_labels)
        self.evaluate(dev_texts, dev_labels, target_names)