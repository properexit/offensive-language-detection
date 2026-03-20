import torch
import torch.nn as nn
from transformers import AutoModel


class MultiTaskBERT(nn.Module):
    """
    Shared BERT encoder with two classification heads:
    Task A: NOT vs OFF
    Task B: targeted vs untargeted (only for OFF)
    """

    def __init__(self, model_name):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_dim = self.encoder.config.hidden_size

        # two separate classification heads
        self.head_a = nn.Linear(hidden_dim, 2)
        self.head_b = nn.Linear(hidden_dim, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # using CLS token representation
        cls_rep = outputs.last_hidden_state[:, 0]

        logits_a = self.head_a(cls_rep)
        logits_b = self.head_b(cls_rep)

        return logits_a, logits_b