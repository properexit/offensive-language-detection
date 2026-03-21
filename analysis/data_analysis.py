import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#english olid training
olid = pd.read_csv("data/raw/english_OLID/olid-training-v1.0.tsv", sep="\t", engine="python", quoting=3)
print("english olid training")
print(f"Total rows:{len(olid)}")
print(f"Columns:{olid.columns.tolist()}")
print("\nLabel Distribution: Task A:")
for val, count in olid["subtask_a"].value_counts().items():
    print(f"{val}: {count}")

b = olid[olid["subtask_b"].notna() & (olid["subtask_b"] != "NULL")]
print("\nLabel Distribution: Task B:")
for val, count in b["subtask_b"].value_counts().items():
    print(f"{val}: {count}")

c = olid[olid["subtask_c"].notna() & (olid["subtask_c"] != "NULL")]
print("\nLabel Distribution: Task C:")
for val, count in c["subtask_c"].value_counts().items():
    print(f"{val}: {count}")

#arabic
tweets, labels = [], []
with open("data/raw/arabic/offenseval-ar-training-v1/offenseval-ar-training-v1.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        tweets.append(parts[1])
        labels.append(parts[2])
ar_df = pd.DataFrame({"tweet": tweets, "label": labels})

print("arabic training")
print(f"Total rows: {len(ar_df)}")
print(f"Columns: {ar_df.columns.tolist()}")
print("\nLabel Distribution: Task A:")
for val, count in ar_df["label"].value_counts().items():
    print(f"{val}: {count}")

#graphs
#graph 1 english
fig1, axes = plt.subplots(1, 3, figsize=(14, 5))
fig1.suptitle("English OLID- Label Distributions", fontsize=14, fontweight="bold")
colors = ["#1E5E9B", "#CBAA35", "#1A9F39"]

ta = olid["subtask_a"].value_counts()
axes[0].bar(ta.index, ta.values, color=colors[:len(ta)])
axes[0].set_title("Task A")
axes[0].set_ylabel("Count")
for i, (k, v) in enumerate(ta.items()):
    axes[0].text(i, v + 50, str(v), ha="center", fontweight="bold")

tb = b["subtask_b"].value_counts()
axes[1].bar(tb.index, tb.values, color=colors[:len(tb)])
axes[1].set_title("Task B")
axes[1].set_ylabel("Count")
for i, (k, v) in enumerate(tb.items()):
    axes[1].text(i, v + 30, str(v), ha="center", fontweight="bold")

tc = c["subtask_c"].value_counts()
axes[2].bar(tc.index, tc.values, color=colors[:len(tc)])
axes[2].set_title("Task C")
axes[2].set_ylabel("Count")
for i, (k, v) in enumerate(tc.items()):
    axes[2].text(i, v + 20, str(v), ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("english_label_distributions.png",dpi=150)

#graph 2 arabic
fig2, ax = plt.subplots(figsize=(6, 5))
fig2.suptitle("Arabic- Label Distribution (Task A)", fontsize=14, fontweight="bold")

ta_ar = ar_df["label"].value_counts()
ax.bar(ta_ar.index, ta_ar.values, color=colors[:len(ta_ar)])
ax.set_title("Task A")
ax.set_ylabel("Count")
for i, (k, v) in enumerate(ta_ar.items()):
    ax.text(i, v + 30, str(v), ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("arabic_label_distributions.png", dpi=150)