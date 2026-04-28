# Retail Customer Churn Prediction

> Système de prédiction du churn client pour le retail, basé sur des modèles de Machine Learning (Random Forest, XGBoost, Gradient Boosting) avec une API REST FastAPI et un dashboard Streamlit.

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-brightgreen?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![License: MIT](https://img.shields.io/badge/Licence-MIT-yellow.svg?style=for-the-badge)

---

## 📋 Table des Matières

- [À Propos](#-à-propos)
- [Architecture du Projet](#-architecture-du-projet)
- [Technologies Utilisées](#-technologies-utilisées)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Pipeline ML](#-pipeline-ml)
- [API REST](#-api-rest)
- [Dashboard Streamlit](#-dashboard-streamlit)
- [Performances des Modèles](#-performances-des-modèles)
- [Roadmap](#-roadmap)
- [Contribution](#-contribution)
- [Licence](#-licence)
- [Contact](#-contact)

---

## 🚀 À Propos

Ce projet implémente un système complet de **prédiction du churn client** pour une entreprise de retail. À partir de données comportementales et transactionnelles (indicateurs RFM, satisfaction, ancienneté, etc.), plusieurs modèles de classification sont entraînés, comparés et déployés via une **API REST** et un **dashboard interactif**.

Le système permet de :
- **Prédire** la probabilité qu'un client quitte l'entreprise dans un futur proche
- **Classifier** le niveau de risque (Low / Medium / High / Critical)
- **Recommander** des actions CRM adaptées à chaque profil
- **Analyser** en batch des portefeuilles clients complets

---

## 🗂 Architecture du Projet

```
projet_ml_retail/
│
├── app/
│   ├── api.py              # API REST FastAPI (endpoints /predict, /predict/batch)
│   ├── app.py              # Dashboard Streamlit (mode standalone)
│   └── app_api.py          # Dashboard Streamlit (mode connecté à l'API)
│
├── data/
│   ├── raw/                # Données brutes (retail_customers_COMPLETE_CATEGORICAL.csv)
│   ├── processed/          # Données prétraitées
│   ├── train_test/         # Splits X_train, X_test, y_train, y_test
│   └── predictions/        # Sorties des prédictions batch
│
├── figures/                # Visualisations (ROC, Confusion Matrix, Feature Importance)
├── models/                 # Artefacts ML sérialisés (.pkl, .joblib)
├── notebooks/              # Notebook d'exploration (exploration_notebook.ipynb)
├── reports/                # Rapports EDA (corrélations, valeurs manquantes, PCA)
│
├── src/
│   ├── preprocessing.py    # Pipeline de prétraitement & feature engineering
│   ├── train_model.py      # Entraînement, évaluation et comparaison des modèles
│   ├── predict.py          # Service de prédiction (PredictionService)
│   └── utils.py            # Fonctions de visualisation & utilitaires
│
├── requirements.txt
├── mk-venv.ps1             # Création de l'environnement virtuel (Windows)
└── README.md
```

---

## 🛠 Technologies Utilisées

| Catégorie | Outils |
|---|---|
| **Langage** | Python 3.10 |
| **ML & Data** | scikit-learn, XGBoost, pandas, numpy, imbalanced-learn |
| **API** | FastAPI, Uvicorn, Pydantic v2 |
| **Dashboard** | Streamlit, Plotly |
| **Visualisation** | Matplotlib, Seaborn |
| **Sérialisation** | joblib |
| **Versioning** | Git / GitHub |

---

## ⚙️ Installation

### Prérequis

- Python 3.10
- Git

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/votre-pseudo/projet_ml_retail.git
cd projet_ml_retail

# 2. Créer et activer l'environnement virtuel (Windows)
python -m venv venv
venv\Scripts\activate

# Ou sur Linux/macOS
python3 -m venv venv
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## 🖥 Utilisation

### 1. Prétraitement des données

```bash
python src/preprocessing.py
```

Génère les fichiers dans `data/processed/` et `data/train_test/`.

### 2. Entraînement des modèles

```bash
cd src
python train_model.py
```

Entraîne et compare Logistic Regression, Random Forest, Gradient Boosting et XGBoost. Les artefacts sont sauvegardés dans `models/`.

### 3. Lancer l'API REST

```bash
# Depuis la racine du projet
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

Documentation interactive disponible sur : [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Lancer le Dashboard Streamlit

```bash
# Mode standalone (sans API)
streamlit run app/app.py

# Mode connecté à l'API (nécessite l'API lancée)
streamlit run app/app_api.py
```

---

## 🔬 Pipeline ML

```
Données brutes (CSV)
        ↓
  Feature Engineering          ← AvgValuePerTransaction, MonetaryCV, ReturnRatioSquared...
        ↓
  Encodage catégoriel           ← Ordinal (AgeCategory, LoyaltyLevel...) + One-Hot (CustomerType, Gender...)
        ↓
  Traitement des IPs            ← Classification Class_A / Class_B / Class_C
        ↓
  Imputation (médiane)
        ↓
  StandardScaler
        ↓
  Train/Test Split (80/20, stratifié)
        ↓
  Entraînement multi-modèles
        ↓
  Voting Classifier (RF + GB + XGBoost)
        ↓
  Sérialisation (best_model.pkl, voting_classifier.pkl)
```

**Features clés utilisées :** Recency, Frequency, MonetaryTotal, TotalQuantity, Age, SupportTicketsCount, SatisfactionScore, ReturnRatio, CustomerTenureDays, UniqueProducts, AvgDaysBetweenPurchases, et variables catégorielles encodées (pays, type client, saison préférée...).

---

## 🌐 API REST

L'API expose les endpoints suivants :

| Méthode | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Statut général de l'API |
| `GET` | `/health` | Healthcheck (monitoring) |
| `GET` | `/model/info` | Métadonnées des modèles chargés |
| `POST` | `/predict` | Prédiction unitaire (1 client) |
| `POST` | `/predict/batch` | Prédiction en lot (N clients) |

**Exemple de requête :**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Recency": 90,
    "Frequency": 2,
    "MonetaryTotal": 150.0,
    "TotalQuantity": 20,
    "Age": 45,
    "SupportTicketsCount": 4,
    "SatisfactionScore": 2,
    "ReturnRatio": 0.25
  }'
```

**Exemple de réponse :**

```json
{
  "churn_prediction": 1,
  "churn_probability": 0.8342,
  "churn_risk_level": "Critical",
  "churn_risk_description": "Risque critique - Intervention urgente",
  "churn_recommendation": "Action urgente: appel commercial, offre spéciale, résolution rapide",
  "timestamp": "2026-04-28T10:35:12.345678"
}
```

---

## 📊 Dashboard Streamlit

Le dashboard propose 4 pages :

- **📝 Prédiction Simple** — Saisie manuelle des données d'un client, jauge de risque interactive
- **📊 Batch Prediction** — Upload CSV, prédictions en masse, téléchargement des résultats
- **📈 Analyse des Risques** — Comparaison des 4 profils types (faible / modéré / élevé / critique)
- **ℹ️ Informations** — Détails sur les modèles chargés et les features importantes

---

## 📈 Performances des Modèles

| Modèle | CV F1-Score | Test F1 | Test AUC |
|---|---|---|---|
| Logistic Regression | — | — | — |
| Random Forest | — | — | — |
| Gradient Boosting | — | — | — |
| XGBoost | — | — | — |
| **Voting Classifier** | — | — | — |

> Les métriques exactes sont générées lors de l'exécution de `train_model.py` et sauvegardées dans `figures/`.

Les courbes ROC, matrices de confusion et importances des features sont disponibles dans le dossier `figures/`.

---

## 📍 Roadmap

- [x] Pipeline de prétraitement complet
- [x] Entraînement et comparaison multi-modèles
- [x] API REST FastAPI avec documentation Swagger
- [x] Dashboard Streamlit (standalone + mode API)
- [x] Prédiction batch avec export CSV
- [ ] Conteneurisation Docker
- [ ] Tests unitaires (pytest)
- [ ] Déploiement cloud (AWS / Azure)
- [ ] Monitoring des dérives de données (data drift)
- [ ] Réentraînement automatique (MLOps)

---

## 🤝 Contribution

Nous suivons le **GitHub Flow** :

1. Forkez le projet.
2. Créez votre branche (`git checkout -b feature/AmazingFeature`).
3. Committez vos modifications (`git commit -m 'Add some AmazingFeature'`).
4. Pushez sur la branche (`git push origin feature/AmazingFeature`).
5. Ouvrez une **Pull Request**.

---

## 📄 Licence

Distribué sous la licence **MIT**. Voir `LICENSE` pour plus d'informations.

---

## ✉️ Contact

**Yassine Missaoui** — yassine.missaoui@enis.tn

Lien du projet : [https://github.com/votre-pseudo/projet_ml_retail](https://github.com/votre-pseudo/projet_ml_retail)
