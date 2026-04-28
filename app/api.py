# app/api.py
import sys
import os
import numpy as np  # <-- AJOUTER CET IMPORT
from datetime import datetime
from typing import Dict, List, Optional

# Ajouter le chemin racine au PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# ============================================================
# CHARGEMENT DIRECT DES MODELES
# ============================================================

# Chemins corrects
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_PATH = os.path.join(BASE_DIR, 'models')

def load_models():
    """Charge les modèles depuis le bon chemin"""
    models = {}
    
    print(f"Recherche des modèles dans: {MODELS_PATH}")
    print("="*60)
    
    model_files = {
        'classifier': 'best_model.pkl',
        'scaler': 'scaler.pkl',
        'feature_names': 'feature_names.pkl',
        'voting_classifier': 'voting_classifier.pkl'
    }
    
    for key, filename in model_files.items():
        filepath = os.path.join(MODELS_PATH, filename)
        if os.path.exists(filepath):
            models[key] = joblib.load(filepath)
            print(f"  ✓ {key} chargé depuis {filename}")
        else:
            print(f"  ✗ {filename} non trouvé dans {MODELS_PATH}")
            models[key] = None
    
    # Vérification
    if models['classifier'] is not None and models['feature_names'] is not None:
        print("\n✅ Modèles chargés avec succès")
        return models, True
    else:
        print("\n❌ Modèles manquants")
        return models, False

# Charger les modèles au démarrage
models, MODELS_LOADED = load_models()

# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Churn Prediction API",
    description="API pour prédire le churn des clients",
    version="1.0.0"
)

# Modèles Pydantic
class CustomerData(BaseModel):
    """Données client pour la prédiction"""
    Recency: int = 50
    Frequency: int = 5
    MonetaryTotal: float = 1000.0
    TotalQuantity: int = 100
    Age: int = 35
    SupportTicketsCount: int = 1
    SatisfactionScore: int = 4
    ReturnRatio: float = 0.05
    AvgDaysBetweenPurchases: float = 30.0
    CustomerTenureDays: int = 200
    UniqueProducts: int = 20

class PredictionResponse(BaseModel):
    """Réponse de prédiction"""
    churn_prediction: int
    churn_probability: float
    churn_risk_level: str
    churn_risk_description: str
    churn_recommendation: str
    timestamp: str

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def get_churn_risk_level(probability: float) -> str:
    """Retourne le niveau de risque"""
    if probability < 0.2:
        return "Low"
    elif probability < 0.4:
        return "Medium"
    elif probability < 0.6:
        return "High"
    else:
        return "Critical"

def get_churn_risk_description(probability: float) -> str:
    """Description du risque"""
    if probability < 0.2:
        return "Faible risque - Client fidèle"
    elif probability < 0.4:
        return "Risque modéré - Surveillance recommandée"
    elif probability < 0.6:
        return "Risque élevé - Actions préventives nécessaires"
    else:
        return "Risque critique - Intervention urgente"

def get_churn_recommendation(probability: float) -> str:
    """Recommandation basée sur le risque"""
    if probability < 0.2:
        return "Maintenir l'engagement, programme de fidélité"
    elif probability < 0.4:
        return "Envoyer des offres personnalisées, suivre l'activité"
    elif probability < 0.6:
        return "Contacter le client, offrir des réductions, enquête de satisfaction"
    else:
        return "Action urgente: appel commercial, offre spéciale, résolution rapide"

def prepare_features(customer_dict: Dict, feature_names: List[str], scaler) -> np.ndarray:
    """Prépare les features pour la prédiction"""
    import pandas as pd
    
    # Créer DataFrame
    df = pd.DataFrame([customer_dict])
    
    # Ajouter les colonnes manquantes
    for feat in feature_names:
        if feat not in df.columns:
            df[feat] = 0
    
    # Garder seulement les colonnes attendues
    df = df[feature_names]
    
    # Standardiser
    X = df.values
    if scaler is not None:
        X = scaler.transform(X)
    
    return X

# ============================================================
# ENDPOINTS API
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Churn Prediction API",
        "status": "running",
        "models_loaded": MODELS_LOADED,
        "models_path": MODELS_PATH
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": MODELS_LOADED,
        "feature_count": len(models.get('feature_names', [])) if models.get('feature_names') else 0
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    """Prédire le churn pour un client"""
    if not MODELS_LOADED:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convertir en dictionnaire
        customer_dict = customer.dict()
        
        # Préparer les features
        X = prepare_features(
            customer_dict, 
            models['feature_names'], 
            models.get('scaler')
        )
        
        # Prédire
        classifier = models['classifier']
        churn_pred = int(classifier.predict(X)[0])
        
        try:
            churn_prob = float(classifier.predict_proba(X)[0, 1])
        except:
            churn_prob = float(churn_pred)
        
        return PredictionResponse(
            churn_prediction=churn_pred,
            churn_probability=churn_prob,
            churn_risk_level=get_churn_risk_level(churn_prob),
            churn_risk_description=get_churn_risk_description(churn_prob),
            churn_recommendation=get_churn_recommendation(churn_prob),
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/batch")
def predict_batch(customers: List[CustomerData]):
    """Prédire le churn pour plusieurs clients"""
    if not MODELS_LOADED:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    results = []
    for i, customer in enumerate(customers):
        try:
            customer_dict = customer.dict()
            X = prepare_features(customer_dict, models['feature_names'], models.get('scaler'))
            
            classifier = models['classifier']
            churn_pred = int(classifier.predict(X)[0])
            
            try:
                churn_prob = float(classifier.predict_proba(X)[0, 1])
            except:
                churn_prob = float(churn_pred)
            
            results.append({
                "index": i,
                "churn_prediction": churn_pred,
                "churn_probability": churn_prob,
                "churn_risk_level": get_churn_risk_level(churn_prob),
                "success": True
            })
        except Exception as e:
            results.append({
                "index": i,
                "error": str(e),
                "success": False
            })
    
    return {"results": results, "total": len(results)}

@app.get("/model/info")
def model_info():
    """Informations sur les modèles"""
    if not MODELS_LOADED:
        return {"models_loaded": False}
    
    return {
        "models_loaded": True,
        "feature_count": len(models.get('feature_names', [])),
        "features_sample": models.get('feature_names', [])[:10],
        "classifier_type": str(type(models.get('classifier'))),
        "scaler_type": str(type(models.get('scaler')))
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)