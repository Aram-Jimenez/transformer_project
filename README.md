# Simplifying Fact-Checking with Transformers: A Study on the LIAR Dataset

## Overview
This project explores the use of Transformer-based models for fact-checking tasks using the LIAR dataset. The LIAR dataset is a challenging benchmark with six nuanced categories for classifying news statements: True, Mostly True, Half True, Barely True, False, and Pants on Fire.

### Problem Statement
The task of fact-checking is inherently difficult due to:
1. **Class Imbalance**: Some categories (e.g., Pants on Fire) are severely underrepresented.
2. **Ambiguity**: Labels like Mostly True and Half True overlap significantly, leading to frequent misclassifications.
3. **Real-World Impact**: Accurate fact-checking models are critical for combating misinformation.

### Approach
To address these challenges, the project:
1. **Simplified Prior Work**: Leveraged Hugging Face's Transformers library to streamline workflows.
2. **Baseline Model**: Fine-tuned a pre-trained BERT model on the LIAR dataset using only statement text.
3. **Improved Model**: Incorporated metadata (speaker, affiliation, context) and implemented weighted loss to address class imbalance.
4. **Fine-Tuned Model**: Applied hyperparameter optimization, including lower learning rates and longer training epochs, for further refinement.

### Summary of Results
The final model achieved comparable performance to previous approaches, emphasizing the effectiveness of simplifications while highlighting the inherent challenges of the LIAR dataset.


## Background and Inspiration
Fact-checking is a critical tool in the fight against misinformation, especially in today's information-driven society. The LIAR dataset has been a standard benchmark for exploring fact-checking models, but prior research has faced several challenges:
- **Complex Labels**: Fact-checking often involves subjective or nuanced interpretations, making it difficult for models to classify statements accurately.
- **Class Imbalance**: Underrepresented categories like "Pants on Fire" skew evaluation metrics and reduce model reliability.
- **Data Limitations**: The dataset is relatively small for a task requiring deep language understanding.

Previous studies using the LIAR dataset, such as the work detailed in the LIAR-PLUS paper, achieved approximately 27%-38% accuracy on this challenging multi-class task. Their approaches involved adding external metadata and sophisticated architectures like Siamese networks to boost performance.

### How This Project Builds on Previous Work
1. **Simplification**:
   - Instead of complex architectures, this project leverages modern tools like Hugging Face's Transformers to streamline the workflow.
   - Metadata is incorporated directly into the input sequence, eliminating the need for additional branches in the model.

2. **Focus on Generalization**:
   - While achieving high accuracy is a challenge due to the dataset's limitations, this project prioritizes clarity, ease of use, and insights from simpler approaches.

3. **Deployability**:
   - A Gradio app is included to demonstrate real-time classification, making the project accessible to users without extensive technical expertise.


