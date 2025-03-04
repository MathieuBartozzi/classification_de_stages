# Méthodologie de classification des stages


## 1. Construction du modèle
L’objectif est de classifier les descriptions de stages en **"observation" (`ok`)** ou **"non observation" (`ko`)**.

## 2. Prétraitement des données
- Suppression des **doublons** et harmonisation des labels (`ok`/`ko`).
- Homogénisation des phrases évaluées plusieurs fois et avec des résultats différents (choix de l'évaluaation la plus fréquente)
- Nettoyage de données : suppressions des caractères spéciaux, des majuscules, des stopwords, des espaces, lemmatisation, correction orthographique
- Tokenisation et vectorisation avec **TF-IDF** (unigrammes).

## 3. Modèles testés
L’entraînement du modèle et l'exploration des données sont documentés dans **`notebooks/text_classification_pipeline.ipynb`**.
Nous avons testé plusieurs modèles :
| Modèle               | Accuracy  | F1-Score  |
|----------------------|-----------|-----------|
| Naïve Bayes         | 0.609848  | 0.524801  |
| Logistic Regression | 0.651515  | 0.608751  |
| SVM                 | 0.655303  | 0.650221  |
| Random Forest       | 0.670455  | 0.631214  |
| **XGBoost**            | **0.670455**  | **0.655317**  |

✅ **XGBoost a été retenu** car il offre le **meilleur compromis** le meilleur score.

---

📉 **Matrice de confusion :**

```
[[134  28]
 [ 59  43]]
```

---

## 4. Améliorations possibles
Voici les **pistes d'amélioration** pour un futur modèle plus performant :

- 📌 **Gérer le déséquilibre des classes** (60% "ko" / 40% "ok")
- 📌 **Optimiser les hyperparamètres** du modele XGBoost (GridSearchCV)
- 📌 **Améliorer le pré-traitement NLP** verifier la qualité de la vectorisation actuelle.
- 📌 **Tester d'autres modèles plus avancés**  `BERT`, ou Réseau de Neurones Récurrent (`RNN`) ou une variante comme un `LSTM` (Long Short-Term Memory).
---

## 5. API & Déploiement
L'API est construite avec **FastAPI** et expose un **endpoint `/predict`**.

1. Assurez-vous d'avoir un environnement virtuel activé et installez les dépendances :

```
pip install -r requirements.txt
```

2. Exécutez la commande suivante depuis le dossier où se trouve votre projet :

```
uvicorn api.api:app --reload
```

L'API tournera alors sur `http://127.0.0.1:8000`.

3. Ouvrir le lien dans et la réponse suivante apparaitra dans votre navigateur :

```
{
  "message": "Bienvenue sur l'API de classification des stages"
}

```

**Exemple d'appel API en local :**
```
curl -X 'POST' 'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"text": "Ce stage permet d’observer un ingénieur."}'

{
  "stage_type": "ok"
}
```
