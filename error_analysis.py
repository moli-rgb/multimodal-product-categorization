import json
import numpy as np
from collections import Counter
from src.dataset.fusion_dataset import load_fusion_data

with open('predictions_for_bootstrap.json') as f:
    data = json.load(f)

_, _, train_labels = load_fusion_data('train')
categories = sorted(set(train_labels))
idx2category = {i: c for i, c in enumerate(categories)}

def top_confusions(preds, labels, idx2category, top_n=10):
    confusions = Counter()
    for pred, label in zip(preds, labels):
        if pred != label:
            confusions[(idx2category[label], idx2category[pred])] += 1
    return confusions.most_common(top_n)

print("Top 10 errors in text model")
for (true_cat, pred_cat), count in top_confusions(data['text_preds'], data['text_labels'], idx2category):
    print(f"  True: {true_cat}    Predicted: {pred_cat}  ({count} times)")

print("\nTop 10 errors in fusion model")
for (true_cat, pred_cat), count in top_confusions(data['fusion_preds'], data['fusion_labels'], idx2category):
    print(f"  True: {true_cat}    Predicted: {pred_cat}  ({count} times)")

text_correct = sum(1 for p, l in zip(data['text_preds'], data['text_labels']) if p == l)
fusion_correct = sum(1 for p, l in zip(data['fusion_preds'], data['fusion_labels']) if p == l)
n = len(data['text_labels'])
print(f"\nOverall accuracy (not macro-F1) - Text: {text_correct/n:.4f}")
print(f"Overall accuracy (not macro-F1) - Fusion: {fusion_correct/n:.4f}")