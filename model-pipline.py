import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def prepare_data(filepath):
    """
    Charge et prétraite les données.
    """
    print(f"📂 Chargement du fichier : {filepath}")
    df = pd.read_csv(filepath)

    # Encodage des variables catégoriques
    label_encoders = {}
    for col in ['State', 'International plan', 'Voice mail plan']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Conversion de la colonne cible 'Churn' en 0 et 1
    df['Churn'] = df['Churn'].astype(int)

    # Suppression des colonnes non pertinentes
    drop_cols = ['Total day minutes', 'Total eve minutes', 'Total night minutes', 'Total intl minutes']
    df.drop(columns=drop_cols, inplace=True)

    # Séparation des features et de la cible
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # Normalisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Séparation en train/test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print("✅ Données préparées avec succès !")
    return X_train, X_test, y_train, y_test, scaler

def train_model(X_train, y_train):
    """
    Entraîne un modèle Random Forest.
    """
    print("🚀 Entraînement du modèle Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    print("✅ Modèle entraîné avec succès !")
    return rf

def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle sur les données de test.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    }
    return metrics

def save_model(model, filename='random_forest_model.pkl'):
    """
    Sauvegarde le modèle et le scaler.
    """
    joblib.dump(model, filename)
    print(f"💾 Modèle sauvegardé sous {filename}")

def load_model(filename='random_forest_model.pkl'):
    """
    Charge un modèle sauvegardé.
    """
    print(f"🔄 Chargement du modèle depuis {filename}...")
    return joblib.load(filename)
