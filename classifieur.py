import joblib

# Charger le modèle
MODEL_PATH = "model/classifier.pkl"
model = joblib.load(MODEL_PATH)

def predict_stage(description: str) -> str:
    """
    Prend une description de stage en entrée et retourne 'ok' ou 'ko'.
    """
    prediction = model.predict([description])
    return prediction[0]

# Test rapide
if __name__ == "__main__":
    test_text = "Le stagiaire observera le travail d'un développeur."
    print(f"Prédiction : {predict_stage(test_text)}")
