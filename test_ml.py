import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, inference, compute_model_metrics

# Sample test data
X = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
y = np.array([0, 1, 0, 1])


def test_train_model_returns_fitted_model():
    """
    Test that `train_model` returns a fitted model with a `predict` method.
    """
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "predict")
    pass


def test_inference_returns_correct_shape():
    """
    Test that `inference` returns predictions with correct shape and values in {0,1}.
    """
    model = train_model(X, y)
    preds = inference(model, X)
    assert preds.shape == y.shape
    assert set(np.unique(preds)).issubset({0, 1})
    pass


def test_compute_model_metrics_returns_all_three_scores():
    """
    Test that `compute_model_metrics` returns precision, recall, and fbeta.
    """
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0
    pass

