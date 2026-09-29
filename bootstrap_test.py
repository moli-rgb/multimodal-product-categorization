import json
import numpy as np
from sklearn.metrics import f1_score

with open('predictions_for_bootstrap.json') as f:
    data = json.load(f)

text_preds = np.array(data['text_preds'])
text_labels = np.array(data['text_labels'])
fusion_preds = np.array(data['fusion_preds'])
fusion_labels = np.array(data['fusion_labels'])

assert list(text_labels) == list(fusion_labels), "Labels do not match! They must represent the same items in the exact same order."

n = len(text_labels)
n_bootstrap = 10000
text_f1s = []
fusion_f1s = []
diffs = []

rng = np.random.default_rng(42)

for i in range(n_bootstrap):
    indices = rng.choice(n, size=n, replace=True)

    sample_labels = text_labels[indices]
    sample_text_preds = text_preds[indices]
    sample_fusion_preds = fusion_preds[indices]

    text_f1 = f1_score(sample_labels, sample_text_preds, average='macro', zero_division=0)
    fusion_f1 = f1_score(sample_labels, sample_fusion_preds, average='macro', zero_division=0)

    text_f1s.append(text_f1)
    fusion_f1s.append(fusion_f1)
    diffs.append(text_f1 - fusion_f1)

diffs = np.array(diffs)

print(f"Mean text macro-F1 across bootstrap samples: {np.mean(text_f1s):.4f}")
print(f"Mean fusion macro-F1 across bootstrap samples: {np.mean(fusion_f1s):.4f}")
print(f"Mean difference (text - fusion): {np.mean(diffs):.4f}")

ci_lower = np.percentile(diffs, 2.5)
ci_upper = np.percentile(diffs, 97.5)
print(f"95% Confidence Interval for the difference: [{ci_lower:.4f}, {ci_upper:.4f}]")

if ci_lower > 0:
    print("Result: Text is significantly better than Fusion (zero is outside the interval)")
elif ci_upper < 0:
    print("Result: Fusion is significantly better than Text (zero is outside the interval)")
else:
    print("Result: The difference is not statistically significant (zero is inside the interval)")