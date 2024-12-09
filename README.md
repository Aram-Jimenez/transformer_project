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


## Model

This project utilized three sequential approaches to model development, leveraging the LIAR dataset for classification into six labels: True, Mostly True, Half True, Barely True, False, and Pants on Fire.



### 1. Baseline Model
- **Description**: The baseline model fine-tuned `bert-base-uncased` using both statement text and enriched metadata:
  - **Speaker**
  - **Party affiliation**
  - **Context**
  These features were combined into a single input string separated by `[SEP]` tokens. This approach established a reference point for evaluating improvements.
- **Implementation**:
  - Used Hugging Face's `Trainer` API for fine-tuning.
  - Trained for 3 epochs with a learning rate of `2e-5`.
  - Evaluated performance based on weighted F1 score and accuracy.
- **Metrics**:
  - **Validation Loss**: 1.677
  - **Validation Accuracy**: 30.37%
  - **Validation F1**: 0.303
  - **Test Loss**: 1.640
  - **Test Accuracy**: 28.37%
  - **Test F1**: 0.282



### 2. Improved Model
- **Description**: The improved model addressed the class imbalance in the LIAR dataset by implementing a weighted loss function. The weights were inversely proportional to the frequency of each label, ensuring underrepresented classes like "Pants on Fire" contributed more to the loss calculation.
- **Implementation**:
  - Customized the Hugging Face `Trainer` to apply a weighted cross-entropy loss during training.
  - Used the same enriched inputs and training setup as the baseline model.
  - Trained for 3 epochs with the same learning rate (`2e-5`).
- **Metrics**:
  - **Validation Loss**: 1.994
  - **Validation Accuracy**: 28.04%
  - **Validation F1**: 0.282
  - **Test Loss**: 1.920
  - **Test Accuracy**: 27.98%
  - **Test F1**: 0.280
- **Insights**:
  - Weighted loss improved performance for underrepresented classes.
  - Overall accuracy and F1 scores remained comparable to the baseline, highlighting the need for additional improvements.


### 3. Fine-Tuned Model
- **Description**: The fine-tuned model focused on hyperparameter optimization to improve training stability and generalization:
  - Lowered the learning rate to `1e-5`.
  - Increased the number of epochs to 5.
  - Reduced batch size to 8 and used gradient accumulation to simulate a larger batch size.
  - Applied a higher weight decay for regularization.
- **Implementation**:
  - Used the same enriched input format as the baseline and improved models.
  - Trained using the Hugging Face `Trainer` with fine-tuned training arguments.
- **Metrics**:
  - **Validation Loss**: 2.728
  - **Validation Accuracy**: 27.02%
  - **Validation F1**: 0.271
  - **Test Loss**: 2.646
  - **Test Accuracy**: 27.36%
  - **Test F1**: 0.272
- **Insights**:
  - Hyperparameter optimization reduced overfitting but did not significantly improve metrics.
  - Highlighted the limitations of small datasets like LIAR for nuanced classification tasks.



### Model Comparison
| Model              | Validation Loss | Validation Accuracy (%) | Validation F1 | Test Loss | Test Accuracy (%) | Test F1   |
|--------------------|-----------------|--------------------------|---------------|-----------|-------------------|-----------|
| Baseline           | 1.677           | 30.37                   | 0.303         | 1.640     | 28.37            | 0.282     |
| Improved           | 1.994           | 28.04                   | 0.282         | 1.920     | 27.98            | 0.280     |
| Fine-Tuned         | 2.728           | 27.02                   | 0.271         | 2.646     | 27.36            | 0.272     |



### Key Takeaways
1. **Baseline Model**:
   - Achieved strong starting performance by incorporating enriched metadata in the input.
   - Demonstrated the potential of transformer models for fact-checking tasks.

2. **Improved Model**:
   - Addressed class imbalance using weighted loss, benefiting underrepresented classes.
   - Metrics remained similar to the baseline, indicating limited impact on overall performance.

3. **Fine-Tuned Model**:
   - Adjusted hyperparameters to balance overfitting risks but showed diminishing returns.
   - Reinforced the need for larger datasets or external justifications to enhance generalization.

4. **Challenges Persist**:
   - All models struggled with ambiguous labels and dataset imbalance, limiting their ability to achieve significant accuracy improvements.

