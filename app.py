from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Charger le modèle
MODEL_PATH = "random_forest.pkl"

if not os.path.exists(MODEL_PATH):
    raise Exception(f"❌ Fichier modèle {MODEL_PATH} non trouvé. Entraînez le modèle avant de lancer l'API.")

model = joblib.load(MODEL_PATH)

# Initialisation de l'application FastAPI
app = FastAPI(title="Churn Prediction API", description="API pour prédire le churn des clients.")

# Définition du format des données d'entrée
class PredictionInput(BaseModel):
    features: list[float]  # Liste des valeurs des features (doit correspondre au modèle entraîné)

@app.post("/predict")
def predict(data: PredictionInput):
    """Effectue une prédiction en utilisant le modèle ML."""
    try:
        # Vérifier que les features sont bien formatées
        features = np.array(data.features).reshape(1, -1)

        # Prédiction avec le modèle
        prediction = model.predict(features)[0]
        probas = model.predict_proba(features)[0].tolist()

        return {
            "prediction": int(prediction),
            "probability": {"churn": probas[1], "no_churn": probas[0]}
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction : {str(e)}")

# Route de test
@app.get("/")
def root():
    return {"message": "🚀 L'API est en ligne ! Utilisez /predict pour effectuer des prédictions."}
