from transformers import logging
logging.set_verbosity_error()

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from datasets.english.loaders_olid import load_task_a_olid


MODEL_PATH = "checkpoints/english_taskA_finetune_lr2e-5_ep3_20260321_1243.pt"
MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


checkpoint = torch.load(MODEL_PATH, map_location="cpu")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

state_dict = checkpoint["model_state_dict"]
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


_, dev_df = load_task_a_olid()
dev_df = dev_df.sample(n=300, random_state=42)

def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()

    return pred, conf


tp, tn, fp, fn = [], [], [], []


for _, row in dev_df.iterrows():
    text = row["tweet"]
    label = int(row["label"])

    pred, conf = predict(text)

    if label == 1 and pred == 1:
        tp.append((text, conf))
    elif label == 0 and pred == 0:
        tn.append((text, conf))
    elif label == 0 and pred == 1:
        fp.append((text, conf))
    elif label == 1 and pred == 0:
        fn.append((text, conf))


def show(name, data):
    print("\n", name)

    for text, conf in data[:1]:
        print(f"Text: {text}")
        print(f"Confidence: {conf:.4f}")
        print()


show("TRUE POSITIVE (OFF vs OFF)", tp)
show("TRUE NEGATIVE (NOT vs NOT)", tn)
show("FALSE POSITIVE (NOT vs OFF)", fp)
show("FALSE NEGATIVE (OFF vs NOT)", fn)