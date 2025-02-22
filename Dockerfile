# Utilisation d'une image Python
FROM python:3.12

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers
COPY . .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port (si besoin)
EXPOSE 8000

# Commande d'exécution
CMD ["python", "main.py", "--train", "--evaluate"]

