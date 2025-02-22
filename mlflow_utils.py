from elasticsearch import Elasticsearch
import time

def get_elasticsearch_client():
    """Connexion à Elasticsearch avec gestion avancée des erreurs"""
    try:
        es = Elasticsearch(["http://localhost:9200"], timeout=10, verify_certs=False)

        if es.ping():
            print("✅ Connexion réussie à Elasticsearch")
            return es
        else:
            print("❌ Impossible de contacter Elasticsearch (ping échoué)")
            return None
    except Exception as e:
        print(f"🚨 Erreur de connexion à Elasticsearch : {e}")
        return None

def log_to_elasticsearch(es, run_id, metric_name, value):
    """Envoie les logs MLflow vers Elasticsearch"""
    if es is None:
        print("⚠️ Elasticsearch non disponible, log ignoré")
        return

    log_data = {
        "run_id": run_id,
        "metric": metric_name,
        "value": value,
        "timestamp": int(time.time() * 1000)  # Format UNIX en millisecondes
    }

    print(f"📤 Tentative d'envoi à Elasticsearch : {log_data}")

    try:
        response = es.index(index="mlflow-metrics", document=log_data)
        print(f"✅ Log envoyé avec succès : {response}")
    except Exception as e:
        print(f"🚨 Erreur lors de l'envoi du log à Elasticsearch : {e}")
