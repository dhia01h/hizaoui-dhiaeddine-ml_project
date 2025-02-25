import argparse
import numpy as np
import os
import mlflow
import mlflow.sklearn
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)
from mlflow.tracking import MlflowClient
from mlflow_utils import (
    get_elasticsearch_client,
    log_to_elasticsearch,
)  # 🔥 Importer Elasticsearch

# 📌 Configuration de MLflow
TRACKING_URI = "http://localhost:5000"  # ✅ Vérifie bien que MLflow tourne sur ce port
EXPERIMENT_NAME = "Churn Prediction Experiment"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

# 📌 Vérifier si l'expérience MLflow existe, sinon la créer
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    print(f"⚠️ Expérience '{EXPERIMENT_NAME}' non trouvée, création en cours...")
    experiment_id = client.create_experiment(EXPERIMENT_NAME)
    print(f"✅ Expérience créée avec ID : {experiment_id}")
else:
    experiment_id = experiment.experiment_id
    print(f"🔄 Expérience existante trouvée : ID {experiment_id}")

mlflow.set_experiment(EXPERIMENT_NAME)

# 📌 Connexion à Elasticsearch
es = get_elasticsearch_client()
if es is None:
    print("⚠️ Elasticsearch est injoignable, les logs ne seront pas envoyés.")


def register_model(run_id, model, X_train):
    """
    Enregistre le modèle dans la Model Registry de MLflow avec un exemple d'entrée.
    """
    model_uri = f"runs:/{run_id}/model"
    input_example = np.array([X_train[0]])

    try:
        mlflow.sklearn.log_model(model, "model", input_example=input_example)
        registered_model = mlflow.register_model(model_uri, "ChurnPredictionModel")
        print(f"✅ Modèle enregistré dans la Model Registry : {registered_model}")
    except Exception as e:
        print(f"❌ Erreur lors de l'enregistrement du modèle : {e}")


def main():
    parser = argparse.ArgumentParser(description="Pipeline Churn avec MLflow")
    parser.add_argument(
        "--data", type=str, default="churn-bigml-80.csv", help="Chemin du fichier CSV"
    )
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle")
    parser.add_argument(
        "--save",
        type=str,
        default="random_forest.pkl",
        help="Nom du fichier pour sauvegarder le modèle",
    )
    parser.add_argument("--load", type=str, help="Charger un modèle sauvegardé")

    args = parser.parse_args()

    print("📂 Chargement des données...")
    X_train, X_test, y_train, y_test, scaler = prepare_data(args.data)
    model = None

    # 📌 Vérifier si le modèle à charger existe
    if args.load and not os.path.exists(args.load):
        print(f"❌ Fichier modèle '{args.load}' non trouvé. Vérifiez le chemin.")
        return

    # 📌 Démarrer un run MLflow
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        mlflow.log_param("data_file", args.data)

        if args.load:
            print(f"🔄 Chargement du modèle depuis {args.load}...")
            model = load_model(args.load)
            if model is None:
                print("❌ Erreur : Impossible de charger le modèle.")
                return
            print("✅ Modèle chargé avec succès !")

        if args.train:
            print("🚀 Entraînement du modèle en cours...")
            model = train_model(X_train, y_train)
            if model is None:
                print("❌ Erreur : L'entraînement du modèle a échoué.")
                return
            print("✅ Modèle entraîné avec succès !")

            register_model(run_id, model, X_train)
            if args.save:
                save_model(model, args.save)
                print(f"💾 Modèle sauvegardé sous {args.save}")

        if args.evaluate:
            if model is None:
                print(
                    "⚠️ Aucun modèle trouvé ! Veuillez en entraîner un ou en charger un."
                )
            else:
                print("📊 Évaluation du modèle en cours...")
                metrics = evaluate_model(model, X_test, y_test)

                for metric, value in metrics.items():
                    mlflow.log_metric(metric, value)
                    print(
                        f"📤 Envoi de la métrique {metric}: {value} à Elasticsearch..."
                    )  # 🔍 Debug
                    try:
                        log_to_elasticsearch(es, run_id, metric, value)
                    except Exception as e:
                        print(f"⚠️ Impossible d'envoyer {metric} à Elasticsearch : {e}")


if __name__ == "__main__":
    main()
