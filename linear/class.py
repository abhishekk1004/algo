import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Load Dataset
data = pd.read_csv("/Users/abhishek/Projects/algorithm/algo/wine.csv")

# Display Dataset
print("First 5 Rows:")
print(data.head())

# Features and Target
# Replace 'target' with your target column name if different

X = data.drop("target", axis=1)
y = data["target"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create KNN Model
knn = KNeighborsClassifier(n_neighbors=5)

# Train Model
knn.fit(X_train, y_train)

# Make Predictions
y_pred = knn.predict(X_test)

# Accuracy
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))