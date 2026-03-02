from sklearn.metrics import f1_score, classification_report


def evaluate_classification(y_true, y_pred, target_names=None):
    """
    Print macro F1 and classification report.
    """

    macro = f1_score(y_true, y_pred, average="macro")

    print("Macro F1:", macro)
    print(classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0
    ))