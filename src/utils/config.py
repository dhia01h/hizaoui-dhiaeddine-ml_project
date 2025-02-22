# 📌 Fichier de configuration globale

# Chemins des données
DATASET_PATH = "churn-bigml-80.csv"

# Configuration du modèle
RANDOM_STATE = 42
N_ESTIMATORS = 100

# MLflow
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "Churn Prediction Experiment"

# Elasticsearch
ELASTICSEARCH_HOST = "http://localhost:9200"
ELASTICSEARCH_INDEX = "mlflow-metrics"
