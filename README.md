# Multimodal Product Categorization

> **Text vs. Image vs. Multimodal Fusion for Amazon Berkeley Objects (ABO)**  
> A deep-learning study investigating whether combining product titles and images actually improves hierarchical product classification.

**Core research question:**  
> **Does fusing text and image signals improve classification performance compared with using text alone?**

---

## Overview

This project studies multimodal product categorization using the **Amazon Berkeley Objects (ABO)** dataset.

Instead of assuming that multimodal learning must outperform single-modality models, the project evaluates the hypothesis experimentally by building and comparing:

- a classical **TF-IDF + SGD** baseline
- a **GRU-based text classifier**
- a **ResNet18 image classifier**
- a **text + image fusion model**

All learned models are evaluated on the same held-out test subset, followed by statistical testing and qualitative error analysis.

The main result is a useful negative finding:

> **On this dataset and experimental setup, multimodal fusion did not improve performance. The text-only model achieved higher macro-F1 than the fusion model, and the difference was statistically significant.**

This makes the project less about simply building a multimodal model and more about **testing whether multimodality actually adds useful predictive information**.

---

## Research Question & Hypothesis

### Question

Does combining product text and product images produce better category classification than using product text alone?

### Hypothesis

Because product titles often contain explicit semantic information about the product and its category, text may provide a stronger signal than images. However, visual information could still provide complementary information for visually distinguishable categories.

The experiment therefore compares the modalities directly rather than assuming the answer beforehand.

---

## Dataset & Task

The project uses the **Amazon Berkeley Objects (ABO)** dataset and focuses on product categorization within its hierarchical category taxonomy.

### Data preparation

The preprocessing pipeline includes:

1. Metadata ingestion using **DuckDB + SQL**
2. Construction of the category hierarchy
3. Sparse-category rollup into parent categories
4. Removal of residual root-level catch-all categories
5. Duplicate handling
6. Price outlier handling
7. English-language filtering
8. Text cleaning
9. Primary-image selection
10. Stratified train/validation/test splitting
11. Split-level deduplication to reduce data leakage

After category rollup, **219 categories** remain.

The final fusion evaluation subset contains approximately:

- **~70K training products**
- **8,821 held-out test products**
- Products with both a title and a primary image

### Leakage prevention

A particularly important part of the pipeline is split-level deduplication.

Products with identical cleaned titles are assigned to the same split rather than being independently distributed across train and test sets. Otherwise, the same listing text could appear in both sets and artificially inflate performance.

---

## Experimental Pipeline

```text
                 Amazon Berkeley Objects
                           │
                           ▼
                 DuckDB + SQL Pipeline
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
       Product           Product          Category
        Text             Images          Hierarchy
          │                │                │
          └────────────┬───┴────────────────┘
                       ▼
                Train / Val / Test
                       │
       ┌───────────────┼────────────────┐
       ▼               ▼                ▼
  Text Baseline    Image Branch     Fusion Branch
   TF-IDF + SGD      ResNet18        GRU + ResNet18
       │               │                │
       └───────────────┼────────────────┘
                       ▼
                 Macro-F1 / Accuracy
                       │
                       ▼
              Bootstrap Significance
                       │
                       ▼
                  Error Analysis
```

---

## Models

| Model | Architecture | Purpose |
|---|---|---|
| **Baseline** | TF-IDF (10K features) + SGDClassifier | Classical sanity-check baseline |
| **Text-only** | Embedding → GRU → Linear | Learn semantic information from product titles |
| **Image-only** | ImageNet-pretrained ResNet18 | Learn visual product features |
| **Fusion** | GRU + ResNet18 → Concatenation → MLP | Test whether image information complements text |

### Text branch

The text model uses a GRU-based sequence classifier with a hidden dimension of **64**.

### Image branch

The image model uses **ImageNet-pretrained ResNet18**, with `layer4` unfrozen while the remaining backbone is frozen.

### Fusion model

The multimodal model combines:

- a GRU text encoder
- a ResNet18 image encoder
- feature projection
- feature concatenation
- an MLP classification head

Different learning rates are used for the pretrained backbone and the remaining model components:

```text
ResNet18 backbone: 1e-5
Other components:  1e-3
```

### An important architectural fix

During development, the projected image features had a much larger numerical range than the GRU representation.

The image features were roughly in the range:

```text
[-100, +200]
```

while the GRU hidden representation was naturally bounded around:

```text
[-1, +1]
```

This scale mismatch caused unstable fusion behavior and extremely large logits.

An explicit `tanh` activation was therefore applied after the image projection to bring the two modalities onto a more comparable scale before concatenation.

This was an important debugging step because the issue was not immediately obvious from the model architecture alone.

---

## Results

All headline results below are **macro-F1 on the same 8,821-product test set**.

| Model | Macro-F1 |
|---|---:|
| TF-IDF + SGD | 0.3800 |
| ResNet18 Image-only | 0.6000 |
| **GRU Text-only** | **0.6956** |
| Fusion | 0.6554 |

### Accuracy comparison

| Model | Accuracy |
|---|---:|
| Text-only | 0.9266 |
| Fusion | 0.9217 |

The accuracy difference is relatively small:

**0.49 percentage points**

while the macro-F1 difference is:

**4.02 percentage points**

This illustrates why macro-F1 was selected as the primary metric. Accuracy is heavily influenced by large and easier categories, whereas macro-F1 gives each category equal weight.

---

## Statistical Significance

The observed difference between the text-only and fusion models was tested using a **paired bootstrap procedure with 10,000 resamples** from the same held-out test products.

| Quantity | Result |
|---|---:|
| Mean text macro-F1 | 0.6822 |
| Mean fusion macro-F1 | 0.6433 |
| Mean difference (text − fusion) | 0.0389 |
| 95% CI | **[0.0139, 0.0637]** |

The confidence interval does not include zero, indicating that the observed text-vs-fusion performance gap is unlikely to be explained by test-set sampling variation alone under this bootstrap procedure.

---

## Error Analysis

The most frequent errors are often **hierarchy-depth errors** rather than completely unrelated category confusions.

Examples include:

```text
Cases & Covers/Back & Bumper Cases
        → Cases & Covers

Women/Jewelry/Necklaces
        ↔ Women/Necklaces

Office & School Supplies
        ↔ Office Supplies
```

In these cases, the model often identifies the general product type but misses the exact position in the category hierarchy.

The category rollup process can also create semantically close parent/child labels, making some boundaries inherently difficult.

### Additional fusion errors

The fusion model also produced visually motivated confusions such as:

```text
Running Shoes ↔ Walking Shoes
LED Bulbs ↔ overlapping category paths
Snack Foods ↔ Pantry Staples
```

These examples suggest that visual similarity can introduce additional ambiguity when visually similar products belong to different semantic categories.

---

## Engineering & Debugging Lessons

Several implementation issues materially affected the validity of intermediate experiments.

### 1. Primary image selection

An early version joined **every image associated with a product**, which could give products with more images disproportionate influence.

This was corrected by using only:

```text
image_order = 1
```

for the primary product image.

### 2. Deterministic vocabulary construction

The dataset-building SQL query did not guarantee stable row ordering.

Because vocabulary indices were assigned according to first appearance, rebuilding the vocabulary could silently change the mapping between words and embedding indices.

This produced severely degraded evaluation results despite the underlying model being healthy.

The issue was fixed by:

- adding an explicit `ORDER BY`
- persisting the training vocabulary
- persisting the category mapping alongside the model checkpoint

### 3. Fusion feature-scale mismatch

The image projection produced features on a much larger scale than the GRU representation.

The explicit `tanh` constraint on the image branch corrected this mismatch and stabilized the fusion representation.

### 4. Training variance

Repeated ResNet18 image-only runs under identical code and hyperparameters produced macro-F1 scores of approximately:

```text
0.42
0.62
```

This ~20-point variation is treated as a limitation rather than hidden or averaged away.

---

## Limitations

This experiment has several important limitations:

- Image-only training showed substantial run-to-run variance.
- The text-vs-fusion comparison uses the **8,821-product subset containing both text and a primary image**, rather than the full text-eligible dataset.
- Category rollup can create ambiguous boundaries between parent and child categories.
- Only **ResNet18** was explored for vision.
- Only a single-layer **GRU** was explored for text.
- Stronger architectures such as **DistilBERT** or larger vision backbones were outside the scope of this iteration.
- A more rigorous study would evaluate multiple random seeds and report aggregate performance and variance.

---

## Key Finding

The central result is not that multimodal learning is ineffective in general.

It is narrower:

> **For this dataset, task, preprocessing pipeline, and model configuration, adding the image branch to the text model reduced macro-F1 rather than improving it.**

The strongest single modality was text:

```text
Text-only   ≈ 0.70 macro-F1
Image-only  ≈ 0.60 macro-F1
Fusion      ≈ 0.66 macro-F1
```

A plausible interpretation is that the text modality contains substantially stronger category-specific information in this setting. When the weaker visual branch is introduced through joint fusion, it can add noise or ambiguous visual cues rather than useful complementary information.

This is an experimental interpretation, not a claim that multimodal fusion is generally inferior.

---

## What This Project Demonstrates

This project goes beyond training several models and comparing scores. It demonstrates an end-to-end experimental workflow involving:

- SQL-based data engineering
- hierarchical label construction
- leakage prevention
- classical ML baselines
- sequence modeling
- transfer learning
- multimodal deep learning
- discriminative learning rates
- metric selection
- statistical significance testing
- error analysis
- reproducibility/debugging
- critical interpretation of negative results

Most importantly, the project treats **"fusion should improve performance" as a hypothesis to test**, not an assumption to confirm.

---

## Future Work

Potential next steps include:

- DistilBERT or another stronger text encoder
- stronger vision backbones
- alternative fusion strategies
- multiple random seeds
- modality ablation studies
- frozen vs. partially fine-tuned encoders
- hierarchical classification objectives
- class-balanced training strategies
- deeper analysis of categories where images provide complementary information

---

## Tech Stack

**Languages & Data**

- Python
- SQL
- DuckDB
- Pandas
- NumPy

**Machine Learning / Deep Learning**

- Scikit-learn
- PyTorch
- GRU
- ResNet18
- Transfer Learning

**Evaluation & Analysis**

- Macro-F1
- Accuracy
- Bootstrap confidence intervals
- Error analysis

---

## Project Structure

```text
.
├── data/
├── notebooks/
├── src/
├── models/
├── reports/
└── README.md
```

> The exact repository structure may evolve as the project develops.

---

## Conclusion

The experiment provides a clear answer for the current setup:

**multimodal fusion did not improve product classification over the text-only model.**

Rather than treating this as a failed multimodal experiment, the result demonstrates why controlled comparisons, leakage prevention, statistical testing, and error analysis matter in applied machine learning research.

A model that is more complex is not automatically a better model.

---

## Author

**Mostafa Ali**

Data Science / Machine Learning

GitHub: **[@moli-rgb](https://github.com/moli-rgb)**

