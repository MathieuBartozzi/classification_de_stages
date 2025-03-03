# Méthodologie de classification des stages



## 1. Construction du modèle
L’objectif est de classifier les descriptions de stages en **"observation" (`ok`)** ou **"non observation" (`ko`)**.

### 🔹 Prétraitement des données
- Suppression des **doublons** et harmonisation des labels (`ok`/`ko`).
- Nettoyage des **phrases suspectes** (ex : phrases vides ou non françaises).
- Tokenisation et vectorisation avec **TF-IDF** (unigrammes).

### 🔹 Modèles testés
Nous avons testé plusieurs modèles :
| Modèle                | Accuracy | F1-score |
|----------------------|----------|----------|
| **Naïve Bayes**      | 0.6729   | 0.6379   |
| **Logistic Regression** ✅ | **0.7030** | **0.6900** |
| **SVM**              | 0.6541   | 0.6514   |
| **Random Forest**    | 0.6992   | 0.6769   |

📌 **Logistic Regression a été retenue** car elle offre le **meilleur compromis** entre performance et rapidité.

---

## 2. Performances du modèle retenu
Le modèle **Logistic Regression** donne les scores suivants :

| **Métrique**          | **Score** |
|----------------------|----------|
| ✅ Accuracy          | 0.7030   |
| ⚖️ Balanced Accuracy | 0.6615   |
| 🎯 Precision        | 0.6979   |
| 🔄 Recall          | 0.7030   |
| 🏆 F1-score        | 0.6900   |

📉 **Matrice de confusion :**

```
[[138  24]
 [ 55  49]]
```

---

## 3. Améliorations possibles
Voici les **pistes d'amélioration** pour un futur modèle plus performant :

- 📌 **Gérer le déséquilibre des classes** (60% "ko" / 40% "ok") → Ajouter une pondération `class_weight='balanced'`
- 📌 **Tester d'autres modèles plus avancés** : `RandomForest`, `XGBoost`, `BERT`
- 📌 **Améliorer le pré-traitement NLP** (lemmatisation, suppression des stopwords)
- 📌 **Tester un ajustement des hyperparamètres** (GridSearch pour `C`, `ngram_range`, etc.)

---

## 4. API & Déploiement
L'API est construite avec **FastAPI** et expose un **endpoint `/predict`**.

**Exemple d'appel API :**
```
curl -X 'POST' 'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"text": "Ce stage permet d’observer un ingénieur."}'

{
  "stage_type": "ok"
}

```
