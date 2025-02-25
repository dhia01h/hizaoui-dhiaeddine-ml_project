import argparse
import mlflow
from training.model_training import train_model
from evaluation.model_evaluation import evaluate_model
from data_processing.data_preparation import prepare_data
from utils.mlflow_utils import get_elasticsearch_client, log_to_elasticsearch, init_mlflow
from utils.config import DATASET_PATH
import joblib  # 📌 Ajout pour sauvegarder et charger le modèle

# Initialisation de MLflow et Elasticsearch
experiment_id = init_mlflow()
es = get_elasticsearch_client()

def main():
    """
    Exécute le pipeline de churn avec MLflow.
    """
    parser = argparse.ArgumentParser(description="Pipeline Churn avec MLflow")
    parser.add_argument("--data", type=str, default=DATASET_PATH, help="Chemin du fichier CSV")
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle")
    parser.add_argument("--save", type=str, help="Nom du fichier pour sauvegarder le modèle")
    parser.add_argument("--load", type=str, help="Charger un modèle sauvegardé")

    args = parser.parse_args()

    # Chargement des données
    X_train, X_test, y_train, y_test, _ = prepare_data(args.data)
    model = None

    # Démarrer une expérience MLflow
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id

        # Charger un modèle existant si demandé
        if args.load:
            print(f"🔄 Chargement du modèle depuis {args.load}...")
            try:
                model = joblib.load(args.load)
                print("✅ Modèle chargé avec succès !")
            except FileNotFoundError:
                print(f"❌ Erreur : Fichier {args.load} introuvable.")
                return

        # Entraîner un modèle si demandé
        if args.train:
            print("🚀 Entraînement du modèle...")
            model = train_model(X_train, y_train)
            print("✅ Modèle entraîné avec succès !")

            # Sauvegarder le modèle si demandé
            if args.save:
                joblib.dump(model, args.save)
                print(f"💾 Modèle sauvegardé sous {args.save}")

        # Évaluer le modèle si demandé
        if args.evaluate:
            if model is None:
                print("⚠️ Aucun modèle trouvé ! Veuillez en entraîner un ou en charger un.")
            else:
                print("📊 Évaluation du modèle en cours...")
                metrics = evaluate_model(model, X_test, y_test)

                # Logger les métriques dans MLflow et Elasticsearch
                for metric, value in metrics.items():
                    mlflow.log_metric(metric, value)
                    log_to_elasticsearch(es, run_id, metric, value)
                print("✅ Évaluation terminée.")

if __name__ == "__main__":
    main()
