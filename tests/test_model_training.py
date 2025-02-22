import pytest
from training.model_training import train_model
import numpy as np

def test_train_model():
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(2, size=100)
    model = train_model(X_train, y_train)
    assert model is not None, "Le modèle ne doit pas être None"
