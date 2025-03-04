import re
import unicodedata
import spacy
import nltk
from nltk.corpus import stopwords
from spellchecker import SpellChecker

# Télécharger les stopwords (si ce n'est pas déjà fait)
nltk.download('stopwords')

# Charger les ressources NLP
nlp = spacy.load("fr_core_news_sm")
STOPWORDS = set(stopwords.words('french'))
spell = SpellChecker(language="fr")

def clean_data(text: str, remove_stopwords=True, correct_spelling=False) -> str:
    """
    Nettoie un texte brut en appliquant :
    - Minuscule, suppression d’accents et ponctuation
    - Lemmatisation avec spaCy
    - Suppression des stopwords avec NLTK (optionnel)
    - Correction orthographique avec pyspellchecker (optionnel)

    Args:
        text (str): Texte brut à nettoyer
        remove_stopwords (bool): Supprimer les stopwords ? (par défaut: True)
        correct_spelling (bool): Corriger l’orthographe ? (par défaut: False)

    Returns:
        str: Texte nettoyé
    """

    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^\w\s]', '', text)

    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_punct]

    if remove_stopwords:
        tokens = [word for word in tokens if word not in STOPWORDS]

    if correct_spelling:
        tokens = [spell.correction(word) if spell.correction(word) else word for word in tokens]

    return " ".join(tokens).strip()
