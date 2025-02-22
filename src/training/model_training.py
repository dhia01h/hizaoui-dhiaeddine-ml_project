from sklearn.ensemble import RandomForestClassifier
from utils.config import RANDOM_STATE, N_ESTIMATORS

def train_model(X_train, y_train):
    """Entraîne un modèle Random Forest"""
    print("🚀 Entraînement du modèle Random Forest...")
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    print("✅ Modèle entraîné avec succès !")
    return model
