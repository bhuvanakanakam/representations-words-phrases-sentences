# Natural Language Processing Assignment

## Overview

This repository contains the code and documentation for the NLP assignment focusing on word similarity, phrase similarity, and sentence similarity. It utilizes various techniques, including co-occurrence matrices, Singular Value Decomposition (SVD), BERT embeddings, and traditional machine learning models. The project evaluates these approaches against standard datasets to understand their effectiveness.

## Project Structure
├── phrase_sentance_similarity.ipynb # Notebook for phrase similarity task
├── sentance_similarity.ipynb # Notebook for sentence similarity task
├── word_similarity.ipynb # Notebook for word similarity task
├── report.md # Detailed report
└── README.md # This file


## Setup Instructions

### Prerequisites

Make sure you have Python 3.8 or higher installed. You can install the necessary libraries using pip:

```
pip install numpy pandas scikit-learn tensorflow transformers torch scipy nltk

````
Dependencies
The following Python libraries are required:

numpy==1.23.1
pandas==1.4.2
scikit-learn==1.0.2
tensorflow==2.10.0
transformers==4.21.0
torch==1.12.1
scipy==1.8.0
nltk==3.7

## Running the Notebooks

### 1. Word Similarity Task

- **Notebook**: `word_similarity.ipynb`
- **Description**: Implements word similarity using co-occurrence matrices and SVD.
- **Steps**:
  1. Preprocess text data from Wikipedia articles.
  2. Construct a co-occurrence matrix.
  3. Apply Singular Value Decomposition (SVD) to reduce dimensionality.
  4. Compute cosine similarity scores and evaluate against the SimLex-999 dataset.

### 2. Phrase Similarity Task

- **Notebook**: `phrase_sentance_similarity.ipynb`
- **Description**: Compares phrase similarity using BERT and GloVe embeddings.
- **Steps**:
  1. Tokenize phrases using the BERT tokenizer.
  2. Generate embeddings with BERT and average word embeddings with GloVe.
  3. Train a Logistic Regression classifier.
  4. Evaluate the model's performance on validation and test sets.

### 3. Sentence Similarity Task

- **Notebook**: `sentance_similarity.ipynb`
- **Description**: Determines sentence similarity using BERT embeddings and the PAWS dataset.
- **Steps**:
  1. Load and preprocess the PAWS dataset.
  2. Tokenize sentences with the BERT tokenizer.
  3. Extract embeddings and average them.
  4. Concatenate sentence embeddings and train a Logistic Regression classifier.
  5. Evaluate model performance on validation and test sets.

## Evaluation Metrics

- **Word Similarity**: Pearson and Spearman Correlation Coefficients
- **Phrase Similarity**: Accuracy, Precision, Recall, F1-Score
- **Sentence Similarity**: Accuracy, Precision, Recall, F1-Score

## Results

### Word Similarity

- **Approach 1**: Co-occurrence matrix + SVD
  - Window Size: 5
  - Pearson Correlation: 0.15224
  - Spearman Correlation: 0.14173

- **Approach 2**: Co-occurrence matrix + SVD with PMI/PPMI
  - Window Size: 5
  - Pearson Correlation: 0.16252
  - Spearman Correlation: 0.19024

### Phrase Similarity

- **BERT Embeddings**: Validation Accuracy: 0.3000, Test Accuracy: 0.2790
- **GloVe Embeddings**: Validation Accuracy: 0.2800, Test Accuracy: 0.2794

### Sentence Similarity

- **Validation Accuracy**: 0.6533
- **Test Accuracy**: 0.6460
- **Precision**: 0.6724
- **Recall**: 0.6295
- **F1-Score**: 0.6506

## Report

For a detailed explanation of methodologies, experimental setups, and findings, please refer to the `pre-corg.pdf` file.


Thank you for exploring this repository. Your feedback is welcome and appreciated!


