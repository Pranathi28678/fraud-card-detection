
# Credit Card Fraud Detection

This repository contains a Jupyter Notebook that demonstrates the process of detecting credit card fraud using machine learning techniques. The notebook covers data preprocessing, model training, evaluation, and prediction.

## Overview

The goal of this project is to build a machine learning model that can accurately detect fraudulent credit card transactions. The dataset used in this project is highly imbalanced, with a small percentage of fraudulent transactions.

## Contents

- `fraud_card_detection.ipynb`: Jupyter Notebook containing the full workflow for credit card fraud detection.
- `data`: Directory containing the dataset used in the project (not included in this repository).

## Dependencies

To run the notebook, you will need the following Python libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

You can install the required libraries using the following command:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage

1. Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/fraud_card_detection.git
```

2. Navigate to the project directory:

```bash
cd fraud_card_detection
```

3. Launch Jupyter Notebook:

```bash
jupyter notebook
```

4. Open the `fraud_card_detection.ipynb` notebook and run the cells sequentially to reproduce the results.

## Data Preprocessing

The notebook includes steps for data preprocessing, such as:

- Handling missing values
- Encoding categorical features
- Normalizing numerical features
- Splitting the dataset into training and testing sets

## Model Training

Several machine learning models are trained and evaluated in the notebook, including:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Machine (SVM)

## Model Evaluation

The models are evaluated using various metrics, such as:

- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC score

## Results

The final model achieves an accuracy of approximately 96%. The notebook includes detailed analysis and visualization of the model's performance.

## Conclusion

The notebook demonstrates a comprehensive approach to building and evaluating a machine learning model for credit card fraud detection. The techniques used can be applied to other imbalanced classification problems.
