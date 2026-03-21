import torch
import re


class ArabicTweetDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for Arabic tweets.
    Only minimal cleaning is applied.
    """

    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.texts = [self._clean(t) for t in list(texts)]
        self.labels = list(labels)

    def _clean(self, text):
        text = str(text)
        # remove retweet marker if present
        text = re.sub(r"^RT\s+@USER:\s*", "", text)
        # reduce multiple mentions
        text = re.sub(r"(@USER\s*){2,}", "@USER ", text)
        # replace special newline token
        text = text.replace("<LF>", " ")
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
        label = int(self.labels[idx])
        assert label in [0, 1], f"Invalid label: {label}"

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }