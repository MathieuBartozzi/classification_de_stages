import joblib
from cleaning import clean_data  # Importer la fonction de nettoyage

# Charger le modèle
MODEL_PATH = "model/classifier.pkl"
model = joblib.load(MODEL_PATH)

# Dictionnaire de correspondance des labels
LABELS = {0: "ko", 1: "ok"}  # Adapté selon ton dataset

def predict_stage(description: str) -> str:
    """
    Prend une description de stage en entrée, applique le nettoyage,
    et retourne 'ok' ou 'ko'.
    """
    cleaned_description = clean_data(description)
    prediction = model.predict([cleaned_description])[0]  # Récupérer la valeur unique
    return LABELS.get(prediction, "inconnu")  # Convertir le label numérique en texte

# Test rapide
if __name__ == "__main__":
    test_text = "Le stagiaire observera le travail d'un développeur."
    print(f"Prédiction : {predict_stage(test_text)}")
