# =============================================================================
# app/api.py
# =============================================================================
# API REST de prédiction du churn client — Projet ML Retail (GI2)
#
# Architecture :
#   - Framework  : FastAPI (ASGI, auto-documentation Swagger/OpenAPI)
#   - Validation : Pydantic v2
#   - ML         : scikit-learn (chargement via joblib)
#   - Serveur    : Uvicorn (ASGI)
#
# Endpoints exposés :
#   GET  /            → Statut général de l'API
#   GET  /health      → Vérification de santé (monitoring)
#   GET  /model/info  → Métadonnées des modèles chargés
#   POST /predict     → Prédiction unitaire (1 client)
#   POST /predict/batch → Prédiction en lot (N clients)
#
# Usage :
#   uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
# =============================================================================

# --- Bibliothèques standard ---------------------------------------------------
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional

# Ajout du répertoire racine du projet au PYTHONPATH pour permettre les imports
# relatifs depuis n'importe quel sous-dossier du projet.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Bibliothèques tierces ----------------------------------------------------
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# =============================================================================
# SECTION 1 — CHARGEMENT DES MODÈLES
# =============================================================================
# Les modèles sont chargés une seule fois au démarrage de l'application
# (pattern "eager loading") pour éviter des latences lors des appels API.
# Tous les artefacts sont stockés dans le dossier `models/` à la racine du projet.
# =============================================================================

# Chemin absolu vers le dossier racine du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Chemin absolu vers le dossier contenant les artefacts ML sérialisés
MODELS_PATH = os.path.join(BASE_DIR, "models")

# Correspondance entre clé interne et nom de fichier sur le disque
MODEL_FILES: Dict[str, str] = {
    "classifier":        "best_model.pkl",       # Modèle principal (ex. RandomForest)
    "scaler":            "scaler.pkl",            # StandardScaler pré-entraîné
    "feature_names":     "feature_names.pkl",     # Liste ordonnée des features attendues
    "voting_classifier": "voting_classifier.pkl", # Ensemble de modèles (optionnel)
}


def load_models() -> tuple[Dict, bool]:
    """
    Charge tous les artefacts ML depuis le disque au démarrage de l'API.

    Processus :
        1. Vérifie l'existence de chaque fichier .pkl dans MODELS_PATH.
        2. Charge les fichiers trouvés via joblib.load().
        3. Retourne un dictionnaire d'artefacts et un booléen indiquant
           si les composants critiques (classifier + feature_names) sont disponibles.

    Returns:
        models (Dict)      : Dictionnaire {clé → artefact chargé | None}.
        is_ready (bool)    : True si l'API est prête à servir des prédictions.

    Raises:
        Ne lève pas d'exception — les erreurs sont loguées et le flag `is_ready`
        est mis à False pour que l'API démarre en mode dégradé.
    """
    models: Dict = {}

    print(f"\n{'='*60}")
    print(f"  Chargement des modèles depuis : {MODELS_PATH}")
    print(f"{'='*60}")

    for key, filename in MODEL_FILES.items():
        filepath = os.path.join(MODELS_PATH, filename)

        if os.path.exists(filepath):
            models[key] = joblib.load(filepath)
            print(f"  ✓ [{key}] chargé  ← {filename}")
        else:
            models[key] = None
            print(f"  ✗ [{key}] ABSENT  — {filepath} introuvable")

    # Les deux artefacts ci-dessous sont indispensables pour toute prédiction.
    is_ready = (models["classifier"] is not None
                and models["feature_names"] is not None)

    status = "✅ API prête" if is_ready else "❌ Composants manquants — mode dégradé"
    print(f"\n  {status}\n{'='*60}\n")

    return models, is_ready


# Chargement global exécuté une seule fois à l'import du module
models, MODELS_LOADED = load_models()


# =============================================================================
# SECTION 2 — INITIALISATION DE L'APPLICATION FASTAPI
# =============================================================================

app = FastAPI(
    title="Churn Prediction API",
    description=(
        "API de prédiction du risque de churn client pour le Projet ML Retail.\n\n"
        "Fournit des prédictions unitaires et en lot à partir d'un modèle "
        "Random Forest entraîné sur des données comportementales (RFM, satisfaction, tenure…)."
    ),
    version="1.0.0",
    contact={
        "name": "Yassine Missaoui",
        "email": "yassine.missaoui@enis.tn",
        
    },
    docs_url="/docs",       # Swagger UI  → http://localhost:8000/docs
    redoc_url="/redoc",     # ReDoc UI    → http://localhost:8000/redoc
)


# =============================================================================
# SECTION 3 — SCHÉMAS PYDANTIC (Validation des entrées / sorties)
# =============================================================================
# Pydantic garantit la validation automatique des types, des valeurs par défaut
# et génère la documentation OpenAPI sans configuration supplémentaire.
# =============================================================================

class CustomerData(BaseModel):
    """
    Données comportementales et démographiques d'un client.

    Toutes les valeurs disposent de valeurs par défaut (client "médian")
    pour faciliter les tests rapides via Swagger UI.
    """

    # --- Indicateurs RFM (Recency / Frequency / Monetary) --------------------
    Recency: int = Field(
        default=50,
        ge=0,
        description="Nombre de jours depuis le dernier achat (plus bas = plus récent)"
    )
    Frequency: int = Field(
        default=5,
        ge=0,
        description="Nombre total de transactions sur la période d'observation"
    )
    MonetaryTotal: float = Field(
        default=1000.0,
        ge=0,
        description="Montant total dépensé en euros sur la période"
    )
    TotalQuantity: int = Field(
        default=100,
        ge=0,
        description="Nombre total d'articles achetés"
    )

    # --- Indicateurs démographiques ------------------------------------------
    Age: int = Field(
        default=35,
        ge=18,
        le=120,
        description="Âge du client en années"
    )

    # --- Indicateurs de satisfaction & support --------------------------------
    SupportTicketsCount: int = Field(
        default=1,
        ge=0,
        description="Nombre de tickets support ouverts"
    )
    SatisfactionScore: int = Field(
        default=4,
        ge=1,
        le=5,
        description="Score de satisfaction client (1 = très insatisfait, 5 = très satisfait)"
    )

    # --- Indicateurs comportementaux ------------------------------------------
    ReturnRatio: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Taux de retour produit (0.0 = aucun retour, 1.0 = tout retourné)"
    )
    AvgDaysBetweenPurchases: float = Field(
        default=30.0,
        ge=0,
        description="Intervalle moyen en jours entre deux achats consécutifs"
    )
    CustomerTenureDays: int = Field(
        default=200,
        ge=0,
        description="Ancienneté client en jours depuis la première commande"
    )
    UniqueProducts: int = Field(
        default=20,
        ge=0,
        description="Nombre de références produits distinctes achetées"
    )

    class Config:
        # Exemples affichés dans Swagger UI
        json_schema_extra = {
            "example": {
                "Recency": 50,
                "Frequency": 5,
                "MonetaryTotal": 1000.0,
                "TotalQuantity": 100,
                "Age": 35,
                "SupportTicketsCount": 1,
                "SatisfactionScore": 4,
                "ReturnRatio": 0.05,
                "AvgDaysBetweenPurchases": 30.0,
                "CustomerTenureDays": 200,
                "UniqueProducts": 20,
            }
        }


class PredictionResponse(BaseModel):
    """
    Résultat enrichi d'une prédiction de churn.

    Inclut non seulement la prédiction brute mais aussi une interprétation
    métier (niveau de risque, description, recommandation d'action).
    """

    churn_prediction: int = Field(
        description="Prédiction binaire : 0 = fidèle, 1 = churn probable"
    )
    churn_probability: float = Field(
        description="Probabilité de churn entre 0.0 et 1.0"
    )
    churn_risk_level: str = Field(
        description="Niveau de risque : Low | Medium | High | Critical"
    )
    churn_risk_description: str = Field(
        description="Description textuelle du niveau de risque"
    )
    churn_recommendation: str = Field(
        description="Action marketing recommandée pour ce niveau de risque"
    )
    timestamp: str = Field(
        description="Horodatage ISO 8601 de la prédiction"
    )


# =============================================================================
# SECTION 4 — FONCTIONS UTILITAIRES MÉTIER
# =============================================================================

def get_churn_risk_level(probability: float) -> str:
    """
    Convertit une probabilité de churn en niveau de risque catégoriel.

    Seuils (calibrés sur les données d'entraînement) :
        < 0.20  → Low      (client fidèle)
        < 0.40  → Medium   (à surveiller)
        < 0.60  → High     (actions préventives)
        ≥ 0.60  → Critical (intervention urgente)

    Args:
        probability (float): Probabilité de churn prédite par le modèle [0, 1].

    Returns:
        str: Niveau de risque parmi {Low, Medium, High, Critical}.
    """
    if probability < 0.20:
        return "Low"
    elif probability < 0.40:
        return "Medium"
    elif probability < 0.60:
        return "High"
    else:
        return "Critical"


def get_churn_risk_description(probability: float) -> str:
    """
    Retourne une description textuelle lisible du niveau de risque.

    Args:
        probability (float): Probabilité de churn prédite [0, 1].

    Returns:
        str: Description compréhensible pour un utilisateur non-technique.
    """
    if probability < 0.20:
        return "Faible risque — Client fidèle, aucune action urgente"
    elif probability < 0.40:
        return "Risque modéré — Surveillance recommandée"
    elif probability < 0.60:
        return "Risque élevé — Actions préventives nécessaires"
    else:
        return "Risque critique — Intervention commerciale urgente"


def get_churn_recommendation(probability: float) -> str:
    """
    Retourne la recommandation d'action marketing adaptée au niveau de risque.

    Args:
        probability (float): Probabilité de churn prédite [0, 1].

    Returns:
        str: Recommandation concrète à destination de l'équipe CRM.
    """
    if probability < 0.20:
        return "Maintenir l'engagement : programme de fidélité, newsletter"
    elif probability < 0.40:
        return "Envoyer des offres personnalisées et suivre l'activité"
    elif probability < 0.60:
        return "Contacter le client, proposer une réduction, lancer une enquête satisfaction"
    else:
        return "Action urgente : appel commercial prioritaire, offre exclusive, résolution rapide"


def prepare_features(
    customer_dict: Dict,
    feature_names: List[str],
    scaler
) -> np.ndarray:
    """
    Transforme un dictionnaire de données client en vecteur de features prêt
    à être consommé par le modèle ML.

    Pipeline :
        1. Conversion du dictionnaire en DataFrame pandas (1 ligne).
        2. Ajout à zéro de toute colonne attendue par le modèle mais absente
           des données reçues (robustesse aux inputs partiels).
        3. Réordonnancement des colonnes selon l'ordre d'entraînement.
        4. Standardisation via le StandardScaler pré-entraîné (si disponible).

    Args:
        customer_dict  (Dict)       : Données brutes du client (clé = feature, valeur = float/int).
        feature_names  (List[str])  : Liste ordonnée des features attendues par le modèle.
        scaler                      : Instance de StandardScaler ou None.

    Returns:
        np.ndarray : Vecteur 2D de shape (1, n_features) prêt pour classifier.predict().

    Raises:
        ValueError : Si feature_names est vide ou None.
    """
    if not feature_names:
        raise ValueError("La liste des features attendues (feature_names) est vide ou None.")

    # 1. Création du DataFrame (1 ligne)
    df = pd.DataFrame([customer_dict])

    # 2. Remplissage à 0 des features absentes de la requête
    missing_features = set(feature_names) - set(df.columns)
    if missing_features:
        for feat in missing_features:
            df[feat] = 0

    # 3. Alignement de l'ordre des colonnes sur celui attendu par le modèle
    df = df[feature_names]

    # 4. Conversion NumPy + standardisation
    X = df.values
    if scaler is not None:
        X = scaler.transform(X)

    return X


# =============================================================================
# SECTION 5 — ENDPOINTS API
# =============================================================================

@app.get(
    "/",
    summary="Statut général de l'API",
    tags=["Monitoring"],
)
def root() -> Dict:
    """
    Endpoint racine — confirme que l'API est démarrée et indique si les
    modèles sont correctement chargés.

    Utile pour un premier test de connectivité (`curl http://localhost:8000/`).
    """
    return {
        "message":       "Churn Prediction API — Projet ML Retail (GI2)",
        "status":        "running",
        "models_loaded": MODELS_LOADED,
        "models_path":   MODELS_PATH,
        "docs":          "/docs",
    }


@app.get(
    "/health",
    summary="Vérification de santé (liveness probe)",
    tags=["Monitoring"],
)
def health() -> Dict:
    """
    Endpoint de healthcheck — utilisé par les orchestrateurs (Kubernetes, Docker)
    pour surveiller la disponibilité du service.

    Retourne :
        - `status`         : "healthy" | "degraded"
        - `models_loaded`  : True si tous les artefacts critiques sont présents
        - `feature_count`  : Nombre de features attendues par le modèle
    """
    feature_count = (
        len(models["feature_names"])
        if models.get("feature_names") is not None
        else 0
    )

    return {
        "status":        "healthy" if MODELS_LOADED else "degraded",
        "models_loaded": MODELS_LOADED,
        "feature_count": feature_count,
    }


@app.get(
    "/model/info",
    summary="Métadonnées des modèles chargés",
    tags=["Modèle"],
)
def model_info() -> Dict:
    """
    Retourne les informations techniques sur les artefacts ML chargés en mémoire :
    type du classifier, type du scaler, nombre et liste partielle des features.

    Utile pour le débogage et la traçabilité des déploiements.
    """
    if not MODELS_LOADED:
        return {"models_loaded": False, "detail": "Modèles non chargés."}

    feature_names = models.get("feature_names", [])

    return {
        "models_loaded":    True,
        "classifier_type":  type(models["classifier"]).__name__,
        "scaler_type":      type(models["scaler"]).__name__ if models.get("scaler") else None,
        "feature_count":    len(feature_names),
        "features_preview": feature_names[:10],  # Aperçu des 10 premières features
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Prédiction unitaire du churn",
    tags=["Prédiction"],
)
def predict(customer: CustomerData) -> PredictionResponse:
    """
    Prédit la probabilité de churn pour **un seul client**.

    Corps de la requête (JSON) : voir le schéma `CustomerData`.

    Retourne :
        - `churn_prediction`       : 0 (fidèle) ou 1 (churn probable)
        - `churn_probability`      : Probabilité continue [0, 1]
        - `churn_risk_level`       : Low | Medium | High | Critical
        - `churn_risk_description` : Interprétation textuelle du risque
        - `churn_recommendation`   : Action CRM recommandée
        - `timestamp`              : Horodatage ISO 8601 de la prédiction

    Codes HTTP :
        200 — Prédiction réussie
        400 — Données client invalides ou erreur de transformation
        503 — Modèles non disponibles (démarrage en mode dégradé)
    """
    # Vérification de disponibilité des modèles
    if not MODELS_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Les modèles ML ne sont pas chargés. Vérifiez le dossier models/."
        )

    try:
        # 1. Sérialisation du modèle Pydantic en dictionnaire Python
        customer_dict = customer.dict()

        # 2. Préparation du vecteur de features (alignement + standardisation)
        X = prepare_features(
            customer_dict,
            feature_names=models["feature_names"],
            scaler=models.get("scaler"),
        )

        # 3. Prédiction binaire (classe 0 ou 1)
        classifier = models["classifier"]
        churn_pred = int(classifier.predict(X)[0])

        # 4. Probabilité de churn (colonne index 1 = classe positive "churn")
        #    Fallback sur la prédiction binaire si predict_proba n'est pas disponible.
        try:
            churn_prob = float(classifier.predict_proba(X)[0, 1])
        except AttributeError:
            churn_prob = float(churn_pred)

        # 5. Construction et retour de la réponse enrichie
        return PredictionResponse(
            churn_prediction=churn_pred,
            churn_probability=round(churn_prob, 4),
            churn_risk_level=get_churn_risk_level(churn_prob),
            churn_risk_description=get_churn_risk_description(churn_prob),
            churn_recommendation=get_churn_recommendation(churn_prob),
            timestamp=datetime.now().isoformat(),
        )

    except Exception as exc:
        # Toute erreur inattendue est renvoyée avec le détail au client (mode dev).
        # En production, remplacer str(exc) par un message générique pour la sécurité.
        raise HTTPException(status_code=400, detail=str(exc))


@app.post(
    "/predict/batch",
    summary="Prédiction en lot (batch)",
    tags=["Prédiction"],
)
def predict_batch(customers: List[CustomerData]) -> Dict:
    """
    Prédit le churn pour **une liste de clients** en un seul appel API.

    Chaque client est traité indépendamment : un échec sur l'un n'interrompt
    pas le traitement des suivants (tolérance aux erreurs partielles).

    Corps de la requête (JSON) : liste de schémas `CustomerData`.

    Retourne un objet `results` contenant pour chaque client :
        - `index`              : Position dans la liste d'entrée (0-indexé)
        - `churn_prediction`   : 0 ou 1
        - `churn_probability`  : Probabilité continue [0, 1]
        - `churn_risk_level`   : Niveau de risque catégoriel
        - `success`            : True si la prédiction a réussi, False sinon
        - `error`              : Message d'erreur (uniquement si success = False)

    Codes HTTP :
        200 — Traitement terminé (même si certaines lignes ont échoué)
        503 — Modèles non disponibles
    """
    if not MODELS_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Les modèles ML ne sont pas chargés. Vérifiez le dossier models/."
        )

    results = []
    classifier = models["classifier"]

    for i, customer in enumerate(customers):
        try:
            customer_dict = customer.dict()

            # Préparation & prédiction
            X = prepare_features(
                customer_dict,
                feature_names=models["feature_names"],
                scaler=models.get("scaler"),
            )
            churn_pred = int(classifier.predict(X)[0])

            try:
                churn_prob = float(classifier.predict_proba(X)[0, 1])
            except AttributeError:
                churn_prob = float(churn_pred)

            results.append({
                "index":             i,
                "churn_prediction":  churn_pred,
                "churn_probability": round(churn_prob, 4),
                "churn_risk_level":  get_churn_risk_level(churn_prob),
                "success":           True,
            })

        except Exception as exc:
            # Enregistrement de l'erreur sans interrompre le batch
            results.append({
                "index":   i,
                "error":   str(exc),
                "success": False,
            })

    success_count = sum(1 for r in results if r["success"])

    return {
        "results":       results,
        "total":         len(results),
        "success_count": success_count,
        "error_count":   len(results) - success_count,
    }


# =============================================================================
# SECTION 6 — POINT D'ENTRÉE DÉVELOPPEMENT
# =============================================================================
# Permet de lancer l'API directement avec : python app/api.py
# En production, utiliser : uvicorn app.api:app --host 0.0.0.0 --port 8000
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="localhost",
        port=8000,
        reload=False,   # Mettre True en développement pour le rechargement automatique
        log_level="info",
    )