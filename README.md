# Simplifying Fact-Checking with Transformers: A Study on the LIAR Dataset

## Overview
Fact-checking has become increasingly critical in combating misinformation. The LIAR dataset, a widely used benchmark for this task, categorizes statements into six nuanced labels:
- **True**
- **Mostly True**
- **Half True**
- **Barely True**
- **False**
- **Pants on Fire**

Each statement is accompanied by metadata such as the speaker, their affiliation, and the context, making it a rich yet challenging dataset.

---

## Problem
Fact-checking presents several key challenges:
1. **Class Imbalance**: Certain categories, like "Pants on Fire," are underrepresented, leading to biased model predictions.
2. **Ambiguous Labels**: Overlapping categories (e.g., "Mostly True" vs. "Half True") often confuse classification models.
3. **Dataset Limitations**: The relatively small size of the LIAR dataset restricts model generalization, especially for nuanced classifications.

---

## Approach
This project builds on prior work, focusing on simplifying workflows while leveraging state-of-the-art Transformer models. We explored three approaches:
1. **Baseline Model**: Fine-tuned BERT using only the statement text.
2. **Improved Model**: Incorporated metadata (e.g., speaker, affiliation) and applied weighted loss to address class imbalance.
3. **Fine-Tuned Model**: Optimized hyperparameters (e.g., learning rates, batch sizes) to improve training stability.

To make the project practical and interactive, we also deployed a Gradio app for real-time classification of statements.

---

## Summary of Findings
- **Performance**: Achieved comparable accuracy to prior research while using simpler, more modern workflows.
- **Metadata**: Incorporating speaker and context metadata improved the performance of underrepresented categories.
- **Challenges**: Dataset imbalance and ambiguous labels remain significant hurdles to achieving high generalization.


