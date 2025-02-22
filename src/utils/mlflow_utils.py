import mlflow
from mlflow.tracking import MlflowClient
from elasticsearch import Elasticsearch
from utils.config import MLFLOW_TRACKING_URI, EXPERIMENT_NAME, ELASTICSEARCH_HOST, ELASTICSEARCH_INDEX

def init_mlflow():
    """Initialise MLflow et vérifie l'existence de l'expérience"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = client.create_experiment(EXPERIMENT_NAME)
        print(f"✅ Expérience MLflow créée : {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        print(f"🔄 Expérience existante : {experiment_id}")

    mlflow.set_experiment(EXPERIMENT_NAME)
    return experiment_id

def get_elasticsearch_client():
    """Connexion à Elasticsearch"""
    try:
        es = Elasticsearch([ELASTICSEARCH_HOST])
        if es.ping():
            print("✅ Connexion réussie à Elasticsearch")
        else:
            print("❌ Échec de connexion à Elasticsearch")
            es = None
    except Exception as e:
        print(f"🚨 Erreur Elasticsearch : {e}")
        es = None
    return es

def log_to_elasticsearch(es, run_id, metric, value):
    """Envoie les logs de MLflow vers Elasticsearch"""
    if es is None:
        print("⚠️ Elasticsearch non disponible, log ignoré")
        return

    doc = {
        "run_id": run_id,
        "metric": metric,
        "value": value,
    }

    try:
        es.index(index=ELASTICSEARCH_INDEX, document=doc)
        print(f"📤 Log envoyé : {doc}")
    except Exception as e:
        print(f"🚨 Erreur lors de l'envoi du log à Elasticsearch : {e}")
