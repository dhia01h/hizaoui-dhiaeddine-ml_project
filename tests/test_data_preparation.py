import pytest
from data_processing.data_preparation import prepare_data

def test_prepare_data():
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    assert X_train.shape[0] > 0, "X_train ne doit pas être vide"
    assert y_train.shape[0] > 0, "y_train ne doit pas être vide"
