# Simplifying Fact-Checking with Transformers: A Study on the LIAR Dataset

## Overview
Fact-checking has become increasingly critical in combating misinformation. The LIAR dataset, a widely used benchmark for this task, categorizes statements into six nuanced labels:
- **True**
- **Mostly True**
- **Half True**
- **Barely True**
- **False**
- **Pants on Fire**

Each statement is accompanied by metadata, including:
- **Speaker**: The person who made the statement.
- **Affiliation**: The speaker's political or organizational ties.
- **Context**: The circumstances surrounding the statement.

This rich dataset presents both opportunities and challenges for building effective fact-checking models.


## Problem
Fact-checking presents several key challenges:
1. **Class Imbalance**: Categories like "Pants on Fire" are underrepresented, leading to biased predictions and lower accuracy for these labels.
2. **Ambiguous Labels**: Overlapping categories (e.g., "Mostly True" vs. "Half True") often confuse classification models.
3. **Dataset Limitations**: The relatively small size of the LIAR dataset restricts model generalization, especially for nuanced classifications.


## Approach
This project builds on prior research while emphasizing simplicity and practicality. Using state-of-the-art Transformer models, we explored three approaches:
1. **Baseline Model**:
   - Fine-tuned a pre-trained BERT model using only the statement text.
   - Established a reference point for further improvements.

2. **Improved Model**:
   - Enriched inputs with metadata (e.g., speaker, affiliation, context).
   - Applied weighted loss to address class imbalance and improve performance for underrepresented categories.

3. **Fine-Tuned Model**:
   - Optimized hyperparameters (e.g., learning rates, batch sizes) for more stable and effective training.
   - Experimented with extended epochs and gradient accumulation.

To demonstrate real-world applicability, the final model was deployed as an interactive Gradio app for real-time predictions.


## Background

### Introduction to the LIAR Dataset
The **LIAR** dataset, introduced in 2017 by William Yang Wang, is a benchmark for fake news classification. It contains:
- **12,836 short statements** labeled into six categories: *true, mostly true, half true, barely true, false, and pants on fire*.
- Metadata accompanying each statement, including:
  - **Speaker**: The person making the statement.
  - **Affiliation**: The speaker's political or organizational ties.
  - **Context**: The circumstances surrounding the statement.

This dataset provides a rich resource for training fact-checking models but poses unique challenges, such as label imbalance and overlapping class definitions.


### Challenges of Fact-Checking
1. **Imbalanced Data**: Labels like "pants on fire" are underrepresented, leading to biased model predictions.
2. **Ambiguous Labels**: Overlaps between categories (e.g., "half true" and "mostly true") make classification difficult.
3. **Small Dataset Size**: The limited amount of data constrains the ability of models to generalize effectively.



### Prior Work
#### The LIAR Dataset Paper (2017)
- Introduced the LIAR dataset as a benchmark for fake news detection.
- Tested traditional machine learning models like Logistic Regression and SVM.
- Achieved an F1 score of **~27.4%** on six-way classification, highlighting the task's difficulty.

#### LIAR-PLUS Dataset (2018)
- Enhanced the LIAR dataset with **justifications** (evidence from fact-checking articles).
- Showed that incorporating justifications improved F1 scores for six-way classification to **37%**.
- Utilized BiLSTM models with dual inputs (claims and justifications), outperforming single-input approaches.

#### Triple Branch Siamese Network (2019)
- Built a three-branch network using BERT to process statements, metadata, and justifications.
- Introduced a "credit score" derived from speaker history to quantify reliability.
- Achieved the highest six-way classification accuracy to date (**37.4%**) on LIAR-PLUS but noted limitations in metadata integration.




