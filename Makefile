# ==============================
# 🔹 Définition des variables
# ==============================
PYTHON = venv/bin/python3
VENV = venv
REQ = requirements.txt
TRAIN_DATA = churn-bigml-80.csv
TEST_DATA = churn-bigml-20.csv
MODEL_FILE = random_forest.pkl
MAIN_SCRIPT = main.py

# ==============================
# 🔹 Vérifications
# ==============================

# Vérifier si l'environnement virtuel est installé
check_venv:
	@if [ ! -d "$(VENV)" ]; then \
		echo "⚠️  Environnement virtuel absent ! Lancez 'make setup'."; \
		exit 1; \
	fi

# Vérifier si le modèle est généré
check_model:
	@if [ ! -f "$(MODEL_FILE)" ]; then \
		echo "⚠️  Modèle non trouvé ! Exécutez 'make train'."; \
		exit 1; \
	fi

# ==============================
# 🔹 Installation et Configuration
# ==============================

# Installation complète
setup:
	@echo "🔄 Création de l'environnement virtuel..."
	@if [ ! -d "$(VENV)" ]; then python3 -m venv $(VENV); fi
	@echo "✅ Environnement virtuel créé."
	@echo "🚀 Installation des dépendances..."
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install -r $(REQ)
	@echo "✅ Dépendances installées."

# Installation rapide
install: check_venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r $(REQ)
	@echo "✅ Dépendances installées."

# ==============================
# 🔹 Lancer Elasticsearch & Kibana
# ==============================

start-stack:
	@echo "🚀 Démarrage d'Elasticsearch & Kibana..."
	docker-compose up -d
	@echo "✅ Elasticsearch & Kibana lancés !"

stop-stack:
	@echo "🛑 Arrêt d'Elasticsearch & Kibana..."
	docker-compose down
	@echo "✅ Stack arrêtée."

check-elasticsearch:
	@echo "🔍 Vérification de l'état d'Elasticsearch..."
	curl -X GET "http://localhost:9200" || echo "❌ Elasticsearch n'est pas accessible."

# ==============================
# 🔹 MLflow
# ==============================

mlflow-ui: check_venv
	@echo "🚀 Démarrage de MLflow UI..."
	$(PYTHON) -m mlflow ui --host 0.0.0.0 --port 5000 &
	@echo "✅ MLflow UI lancé sur http://localhost:5000"

# ==============================
# 🔹 Machine Learning Pipeline
# ==============================

prepare: check_venv
	$(PYTHON) $(MAIN_SCRIPT) --data $(TRAIN_DATA)

train: check_venv
	$(PYTHON) $(MAIN_SCRIPT) --data $(TRAIN_DATA) --train --save $(MODEL_FILE)

evaluate: check_venv check_model
	$(PYTHON) $(MAIN_SCRIPT) --data $(TEST_DATA) --evaluate --load $(MODEL_FILE)

# ==============================
# 🔹 Vérifications CI/CD
# ==============================

lint: check_venv
	$(PYTHON) -m pylint $(MAIN_SCRIPT) model_pipeline.py

format: check_venv
	$(PYTHON) -m black $(MAIN_SCRIPT) model_pipeline.py

test: check_venv
	PYTHONPATH=. $(PYTHON) -m pytest tests/

security: check_venv
	$(PYTHON) -m bandit -r model_pipeline.py

ci: check_venv lint format test security

# ==============================
# 🔹 Lancer l'API FastAPI
# ==============================

run-api: check_venv check_model
	$(PYTHON) -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

# ==============================
# 🔹 Nettoyage
# ==============================

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache .mypy_cache $(MODEL_FILE)
	@echo "🧹 Nettoyage effectué."

# ==============================
# 🔹 Aide
# ==============================

help:
	@echo "Commandes disponibles :"
	@echo "  make setup          -> Installer l'environnement"
	@echo "  make install        -> Installer les dépendances"
	@echo "  make start-stack    -> Lancer Elasticsearch & Kibana"
	@echo "  make stop-stack     -> Arrêter Elasticsearch & Kibana"
	@echo "  make check-elasticsearch -> Vérifier si Elasticsearch est actif"
	@echo "  make mlflow-ui      -> Démarrer MLflow UI"
	@echo "  make prepare        -> Préparer les données"
	@echo "  make train          -> Entraîner le modèle"
	@echo "  make evaluate       -> Évaluer le modèle"
	@echo "  make lint           -> Vérifier la qualité du code"
	@echo "  make format         -> Reformater le code"
	@echo "  make test           -> Exécuter les tests"
	@echo "  make security       -> Vérifier la sécurité"
	@echo "  make ci             -> Exécuter toutes les étapes CI"
	@echo "  make run-api        -> Lancer l'API"
	@echo "  make clean          -> Nettoyer les fichiers inutiles"
