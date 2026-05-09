# KNN Folder

This folder contains a K-Nearest Neighbors classification example built on Titanic survival data.

## What is included

- `knn.py`: A KNN classifier that predicts whether a passenger survived using selected Titanic features.

## Main topics covered

- Loading a CSV dataset with pandas
- Selecting features and target labels
- Cleaning missing values
- Encoding categorical variables
- Splitting data into train, validation, and test sets
- Feature scaling with `StandardScaler`
- Choosing the best `k` value with validation accuracy
- Training the final KNN model
- Evaluating with accuracy and classification report

## Requirements

Install these Python packages before running the script:

- `pandas`
- `scikit-learn`
- `seaborn`

## How to run

From the project root or the `knn` folder, run:

```bash
python knn.py
```

## Step-by-step workflow

1. Load the Titanic dataset.
2. Keep the required columns.
3. Drop rows with missing values.
4. Encode `sex` and `embarked` into numeric values.
5. Split the data into training, validation, and testing sets.
6. Scale the features.
7. Try multiple `k` values and measure validation accuracy.
8. Pick the best `k` and train the final model.
9. Evaluate the model on the test set.

## KNN concepts to know

- Distance-based learning
- Importance of feature scaling
- Effect of the `k` parameter
- Validation set selection
- Classification metrics

## Expected output

The script prints:

- The first rows of the dataset
- Validation accuracy for each `k` from 1 to 11
- The best `k` value
- Final test accuracy
- Classification report

## Notes

- `knn.py` currently reads data from `/Users/sumanshrestha/Downloads/titanic.csv`. Update that path if the dataset is stored somewhere else.
- The `seaborn` import is present in the script, but it is not used.

## Learning goal

This folder is useful for understanding how KNN works in a real classification problem and how validation helps choose a better `k`.