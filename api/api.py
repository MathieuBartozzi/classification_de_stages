from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import classifier  # Importation du classifieur
import logging

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation de l'API
app = FastAPI(title="Stage Classifier API", description="API qui classe les descriptions de stage")

class StageDescription(BaseModel):
    text: str

# Endpoint de test
@app.get("/")
def root():
    return {"message": "Bienvenue sur l'API de classification des stages"}

# Endpoint de prédiction
@app.post("/predict")
def predict_stage_endpoint(description: StageDescription):
    """
    Endpoint API qui prend une description de stage et renvoie si c'est un stage d'observation ('ok') ou non ('ko').
    """
    logger.info("Requête reçue avec texte")  # Ajout du log

    text = description.text.strip()

    if not text:
        logger.warning("Requête avec texte vide")
        raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide.")

    prediction = classifier.predict_stage(text)
    logger.info(f"Prédiction reçue : {text} → {prediction}")

    return {"stage_type": prediction}


# Commande pour lancer l'API localement :
# uvicorn api.api:app --reload
