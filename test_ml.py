import os
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from ml.model import compute_model_metrics, inference, load_model, save_model

def test_compute_model_metrics():
    """
    Test that compute_model_metrics correctly calculates precision, 
    recall, and F-beta score, returning floats within [0, 1].
    """
    y_true = np.array([1, 1, 0, 0])
    preds = np.array([1, 0, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y_true, preds)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)

    assert precision == 1.0
    assert recall == 0.5


def test_inference():
    """
    Checks that the predictions return an array with the same number of items as rows in X.
    """
    X = np.array([[1, 2], [3, 4]])
    
    # 1. Store the trained model
    model = LogisticRegression().fit(X, [0, 1])

    # 2. Store the predictions returned by inference
    preds = inference(model, X)

    # 3. Assert correct length
    assert len(preds) == len(X)


def test_save_model_and_load_model(tmp_path):
    """
    Checks that the loaded object is not None and has the expected type or attributes.
    """
    dummy_model = LogisticRegression()
    dummy_model.fit([[1, 2], [3, 4]], [0, 1])

    file_path = str(tmp_path / "test_model.pkl")

    save_model(dummy_model, file_path)
    loaded_model = load_model(file_path)

    assert os.path.exists(file_path)
    assert loaded_model is not None
    assert type(loaded_model) == type(dummy_model)
