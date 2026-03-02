import torch
import re


class TweetDataset(torch.utils.data.Dataset):
    """
    Basic dataset for English tweets.
    Includes a small amount of cleaning before tokenization.
    """

    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len

        # convert to lists and clean text
        self.texts = [self._clean(t) for t in list(texts)]
        self.labels = list(labels)

    def _clean(self, text):
        text = str(text)
        # remove RT prefix
        text = re.sub(r"^RT\s+@USER:\s*", "", text)
        # collapse repeated mentions
        text = re.sub(r"(@USER\s*){2,}", "@USER ", text)
        # replace special newline token
        text = text.replace("<LF>", " ")
        # remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }