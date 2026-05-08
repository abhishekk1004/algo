# Linear Folder

This folder contains simple machine learning examples built around the K-Nearest Neighbors (KNN) algorithm. The scripts show the full workflow from loading data to training, prediction, and evaluation.

## What is included

- `linear.py`: KNN classification using the built-in Iris dataset from scikit-learn.
- `class.py`: KNN classification using a CSV dataset loaded with pandas.

## Main topics covered

- Dataset loading
- Train and test split
- Feature scaling with `StandardScaler`
- KNN model creation with `KNeighborsClassifier`
- Model training and prediction
- Accuracy measurement
- Confusion matrix
- Classification report

## Requirements

Install these Python packages before running the scripts:

- `pandas`
- `scikit-learn`

If you are using a virtual environment, activate it first.

## How to run

From the `linear` folder, run:

```bash
python linear.py
```

To run the CSV-based example:

```bash
python class.py
```

## Step-by-step workflow

1. Load the dataset.
2. Separate features and target labels.
3. Split the data into training and testing sets.
4. Scale the input features.
5. Create the KNN classifier.
6. Fit the model on training data.
7. Predict on test data.
8. Evaluate the results.

## KNN concepts to know

- Distance-based learning
- Choice of `k`
- Effect of feature scaling
- Classification versus regression
- Overfitting and underfitting
- Evaluation metrics

## Expected output

The scripts print:

- Predicted values
- Accuracy score
- Confusion matrix
- Classification report

## Notes

- `linear.py` uses `load_iris()` and does not need a CSV file.
- `class.py` currently points to `/Users/abhishek/Projects/algorithm/algo/wine.csv`, but the folder contains `wine-clustering.csv`. If you want to run it successfully, update the file path or rename the dataset file.

## Learning goal

This folder is useful for understanding the basic KNN pipeline and how preprocessing changes model performance.