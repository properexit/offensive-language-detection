import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

from models.multitask_model import MultiTaskBERT
from datasets.english.loaders import load_task_a_english, load_task_b_english
from datasets.english.dataset import TweetDataset
from utils.device import get_device


def train_multitask(
    model_name="google/bert_uncased_L-2_H-128_A-2",
    epochs=3,
    batch_size=16,
    lr=2e-5,
    lambda_b=0.5
):
    """
    Simple hierarchical multi-task training for English.
    Task A = offensive detection
    Task B = targeted vs untargeted (only when offensive)
    """

    device = get_device()
    print("Using device:", device)

    # load datasets
    train_a, dev_a = load_task_a_english()
    train_b, _ = load_task_b_english()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_a = train_a.reset_index(drop=True)
    train_b = train_b.reset_index(drop=True)
    dev_a = dev_a.reset_index(drop=True)

    # build mapping from text → Task B label
    b_label_map = dict(zip(train_b["tweet"], train_b["label"]))

    train_dataset = TweetDataset(train_a["tweet"], train_a["label"], tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = MultiTaskBERT(model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    print("Starting multi-task training")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:

            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_a = batch["labels"].to(device)

            # get cleaned texts directly
            batch_texts = batch["texts"]

            logits_a, logits_b = model(input_ids, attention_mask)

            # Task A loss
            loss_a = loss_fn(logits_a, labels_a)

            # Task B only for OFF samples
            off_mask = labels_a == 1

            if off_mask.sum() > 0:

                labels_b = []

                for text, is_off in zip(batch_texts, off_mask.cpu()):
                    if is_off:
                        labels_b.append(b_label_map.get(text, 0))

                labels_b = torch.tensor(labels_b).to(device)

                loss_b = loss_fn(logits_b[off_mask], labels_b)

                loss = loss_a + lambda_b * loss_b
            else:
                loss = loss_a

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print("Epoch", epoch + 1, "loss:", round(total_loss, 4))

    # evaluation (Task A)
    model.eval()
    preds = []
    gold = []

    dev_dataset = TweetDataset(dev_a["tweet"], dev_a["label"], tokenizer)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits_a, _ = model(input_ids, attention_mask)
            predictions = torch.argmax(logits_a, dim=1)

            preds.extend(predictions.cpu().numpy())
            gold.extend(labels.cpu().numpy())

    print("Task A macro F1:",
          f1_score(gold, preds, average="macro"))