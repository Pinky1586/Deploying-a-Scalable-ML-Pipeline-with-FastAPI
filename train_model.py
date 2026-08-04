import os

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)
#load the cencus.csv data
project_path = "."
data_path = os.path.join(project_path, "data", "census.csv")
print(data_path)
data = pd.read_csv(data_path)

#Implementing data split and k-fold cross validation

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# 1. Load census.csv data
project_path = "."
data_path = os.path.join(project_path, "data", "census.csv")
print(f"Loading data from: {data_path}")
data = pd.read_csv(data_path)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# 2. Train/Test split
train, test = train_test_split(data, test_size=0.2, random_state=42)

# 3. Process training data (Creates X_train, y_train, fitted encoder, and lb)
X_train, y_train, encoder, lb = process_data(
    train, 
    categorical_features=cat_features, 
    label="salary", 
    training=True,
)

# 4. Process test data (Reuses fitted encoder and lb)
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# 5. Train model on processed training data
model = train_model(X_train, y_train)

# 6. Save model and encoder artifacts
model_dir = os.path.join(project_path, "model")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "model.pkl")
encoder_path = os.path.join(model_dir, "encoder.pkl")
save_model(model, model_path)
save_model(encoder, encoder_path)

# 7. Load model back to verify artifact persistence
model = load_model(model_path)

# 8. Run inference and evaluate overall test set performance
preds = inference(model, X_test)
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Overall Metrics -> Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# 9. Compute slice metrics and save output cleanly to slice_output.txt
slice_file_path = os.path.join(project_path, "slice_output.txt")
with open(slice_file_path, "w") as f:
    for col in cat_features:
        for slicevalue in sorted(test[col].unique()):
            count = test[test[col] == slicevalue].shape[0]
            p_slice, r_slice, f1_slice = performance_on_categorical_slice(
                data=test, 
                column_name=col, 
                slice_value=slicevalue, 
                categorical_features=cat_features, 
                label="salary", 
                encoder=encoder, 
                lb=lb, 
                model=model
            )
            f.write(f"{col}: {slicevalue}, Count: {count:,}\n")
            f.write(f"Precision: {p_slice:.4f} | Recall: {r_slice:.4f} | F1: {f1_slice:.4f}\n\n")

print(f"Slice performance successfully written to {slice_file_path}")
