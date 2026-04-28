"""
predict.py - Service de prédiction pour FastAPI et Streamlit
Version adaptée aux modèles existants (best_model.pkl, scaler.pkl, feature_names.pkl)
Fonctionnalités:
- Prédiction simple et batch
- API pour FastAPI
- Interface pour Streamlit
- Export des résultats
"""

import pandas as pd
import numpy as np
import sys
import os
import json
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import joblib


# ============================================================
# CLASSE PRINCIPALE DE PREDICTION
# ============================================================

class PredictionService:
    """Service centralisé de prédiction pour FastAPI et Streamlit"""
    
    def __init__(self, models_path: str = '../models'):
        """
        Initialise le service de prédiction
        
        Args:
            models_path: Chemin vers le dossier contenant les modèles
        """
        self.models_path = models_path
        self.models = {}
        self.is_loaded = False
        self.model_info = {}
        
    def load_all_models(self) -> Dict:
        """
        Charge tous les modèles nécessaires
        
        Returns:
            Dictionnaire des modèles chargés
        """
        print("\n" + "="*60)
        print("CHARGEMENT DES MODELES")
        print("="*60)
        
        # Liste des modèles à charger avec leurs noms de fichiers
        model_files = {
            'scaler': 'scaler.pkl',
            'feature_names': 'feature_names.pkl',
            'classifier': 'best_model.pkl',
            'voting_classifier': 'voting_classifier.pkl',
            'label_encoder_country': 'label_encoder_country.joblib',
            'preprocessing_pipeline': 'preprocessing_pipeline.joblib',
            'standard_scaler': 'standard_scaler.joblib'
        }
        
        loaded_count = 0
        for model_key, filename in model_files.items():
            try:
                full_path = os.path.join(self.models_path, filename)
                if os.path.exists(full_path):
                    self.models[model_key] = joblib.load(full_path)
                    print(f"  ✓ {model_key} chargé depuis {filename}")
                    loaded_count += 1
                else:
                    print(f"  ⚠️ Fichier non trouvé: {filename}")
                    self.models[model_key] = None
            except Exception as e:
                print(f"  ✗ Erreur chargement {filename}: {e}")
                self.models[model_key] = None
        
        # Vérification des modèles critiques
        required_models = ['classifier', 'feature_names']
        missing = [m for m in required_models if self.models.get(m) is None]
        
        if missing:
            print(f"\n⚠️ Modèles manquants: {missing}")
            print("Exécutez d'abord train_model.py")
            self.is_loaded = False
        else:
            self.is_loaded = True
            print(f"\n✅ {loaded_count} modèles chargés avec succès")
            print(f"   Features attendues: {len(self.models['feature_names'])}")
            
            # Stocker les infos des modèles
            self.model_info = {
                'feature_count': len(self.models['feature_names']),
                'features_list': self.models['feature_names'][:20],  # Top 20 seulement
                'models_loaded': [k for k, v in self.models.items() if v is not None],
                'load_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        return self.models
    
    def get_churn_risk_description(self, probability: float) -> str:
        """
        Description du risque de churn
        
        Args:
            probability: Probabilité de churn (0-1)
            
        Returns:
            Description textuelle du risque
        """
        if probability < 0.2:
            return "Faible risque - Client fidèle"
        elif probability < 0.4:
            return "Risque modéré - Surveillance recommandée"
        elif probability < 0.6:
            return "Risque élevé - Actions préventives nécessaires"
        else:
            return "Risque critique - Intervention urgente"
    
    def get_churn_risk_level(self, probability: float) -> str:
        """
        Retourne le niveau de risque (Low/Medium/High/Critical)
        
        Args:
            probability: Probabilité de churn (0-1)
            
        Returns:
            Niveau de risque
        """
        if probability < 0.2:
            return "Low"
        elif probability < 0.4:
            return "Medium"
        elif probability < 0.6:
            return "High"
        else:
            return "Critical"
    
    def get_churn_recommendation(self, probability: float) -> str:
        """
        Recommandation basée sur le risque de churn
        
        Args:
            probability: Probabilité de churn (0-1)
            
        Returns:
            Recommandation d'action
        """
        if probability < 0.2:
            return "Maintenir l'engagement, programme de fidélité"
        elif probability < 0.4:
            return "Envoyer des offres personnalisées, suivre l'activité"
        elif probability < 0.6:
            return "Contacter le client, offrir des réductions, enquête de satisfaction"
        else:
            return "Action urgente: appel commercial, offre spéciale, résolution rapide"
    
    def prepare_features(self, customer_data: Dict[str, Any]) -> np.ndarray:
        """
        Prépare les features pour la prédiction
        
        Args:
            customer_data: Dictionnaire des features du client
            
        Returns:
            Array numpy des features préparées
        """
        if not self.is_loaded:
            raise ValueError("Modèles non chargés. Appelez load_all_models() d'abord")
        
        # Créer DataFrame
        df = pd.DataFrame([customer_data])
        
        # Aligner avec les features attendues
        expected_features = self.models['feature_names']
        if expected_features is not None:
            # Ajouter les colonnes manquantes avec 0
            for feat in expected_features:
                if feat not in df.columns:
                    df[feat] = 0
            # Garder seulement les colonnes attendues dans l'ordre
            df = df[expected_features]
        
        X = df.values
        
        # Appliquer le scaler si disponible
        scaler = self.models.get('scaler')
        if scaler is not None:
            X = scaler.transform(X)
        
        return X
    
    def predict_churn(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prédiction du churn
        
        Args:
            X: Features préparées
            
        Returns:
            Tuple (predictions, probabilités)
        """
        if not self.is_loaded or self.models.get('classifier') is None:
            raise ValueError("Classifier non chargé")
        
        classifier = self.models['classifier']
        predictions = classifier.predict(X)
        
        try:
            probabilities = classifier.predict_proba(X)[:, 1]
        except (AttributeError, IndexError):
            probabilities = predictions.astype(float)
        
        return predictions, probabilities
    
    def predict_single_customer(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prédiction pour un client unique
        
        Args:
            customer_data: Dictionnaire des features du client
            
        Returns:
            Dictionnaire avec toutes les prédictions et métadonnées
        """
        if not self.is_loaded:
            raise ValueError("Modèles non chargés. Appelez load_all_models() d'abord")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'customer_data': customer_data.copy(),
            'success': False
        }
        
        try:
            X = self.prepare_features(customer_data)
            results['features_prepared'] = True
            results['feature_shape'] = X.shape
        except Exception as e:
            results['error'] = f"Erreur préparation features: {e}"
            results['features_prepared'] = False
            return results
        
        # Churn prediction avec best_model
        try:
            churn_pred, churn_prob = self.predict_churn(X)
            results['churn_prediction'] = int(churn_pred[0])
            results['churn_probability'] = float(churn_prob[0])
            results['churn_risk'] = self.get_churn_risk_description(churn_prob[0])
            results['churn_risk_level'] = self.get_churn_risk_level(churn_prob[0])
            results['churn_recommendation'] = self.get_churn_recommendation(churn_prob[0])
            results['success'] = True
        except Exception as e:
            results['churn_error'] = str(e)
            results['churn_prediction'] = -1
            results['churn_probability'] = -1
        
        # Prédiction avec voting classifier si disponible
        voting_clf = self.models.get('voting_classifier')
        if voting_clf is not None:
            try:
                voting_pred = voting_clf.predict(X)
                results['voting_prediction'] = int(voting_pred[0])
                
                # Vérifier si les prédictions concordent
                if 'churn_prediction' in results and results['churn_prediction'] == results['voting_prediction']:
                    results['predictions_agree'] = True
                else:
                    results['predictions_agree'] = False
            except Exception as e:
                results['voting_error'] = str(e)
        
        return results
    
    def predict_batch(self, customers_list: List[Dict[str, Any]], 
                      include_details: bool = False) -> pd.DataFrame:
        """
        Prédiction batch pour plusieurs clients
        
        Args:
            customers_list: Liste de dictionnaires des features
            include_details: Inclure les détails supplémentaires
            
        Returns:
            DataFrame avec les prédictions
        """
        if not self.is_loaded:
            raise ValueError("Modèles non chargés. Appelez load_all_models() d'abord")
        
        results = []
        for i, customer in enumerate(customers_list):
            pred = self.predict_single_customer(customer)
            
            # Créer un résultat simplifié pour la batch
            simplified = {
                'index': i,
                'churn_prediction': pred.get('churn_prediction', -1),
                'churn_probability': pred.get('churn_probability', -1),
                'churn_risk_level': pred.get('churn_risk_level', 'Unknown'),
                'success': pred.get('success', False)
            }
            
            if include_details:
                simplified['voting_prediction'] = pred.get('voting_prediction', -1)
                simplified['predictions_agree'] = pred.get('predictions_agree', False)
            
            results.append(simplified)
        
        return pd.DataFrame(results)
    
    def predict_from_dataframe(self, df: pd.DataFrame, 
                                customer_id_col: str = None) -> pd.DataFrame:
        """
        Prédiction à partir d'un DataFrame
        
        Args:
            df: DataFrame contenant les features
            customer_id_col: Nom de la colonne contenant l'ID client (optionnel)
            
        Returns:
            DataFrame avec les prédictions ajoutées
        """
        # Sauvegarder les IDs si présents
        ids = None
        if customer_id_col and customer_id_col in df.columns:
            ids = df[customer_id_col].copy()
            df_features = df.drop(columns=[customer_id_col])
        else:
            df_features = df.copy()
        
        # Convertir en liste de dictionnaires
        customers_list = df_features.to_dict(orient='records')
        
        # Prédire
        predictions_df = self.predict_batch(customers_list, include_details=True)
        
        # Construire le résultat
        result_df = df.copy()
        
        # Ajouter les prédictions
        result_df['churn_prediction'] = predictions_df['churn_prediction'].values
        result_df['churn_probability'] = predictions_df['churn_probability'].values
        result_df['churn_risk_level'] = predictions_df['churn_risk_level'].values
        
        if 'voting_prediction' in predictions_df.columns:
            result_df['voting_prediction'] = predictions_df['voting_prediction'].values
            result_df['predictions_agree'] = predictions_df['predictions_agree'].values
        
        return result_df
    
    def export_predictions(self, predictions: pd.DataFrame, 
                          output_path: str = None) -> str:
        """
        Exporte les prédictions vers un fichier CSV
        
        Args:
            predictions: DataFrame des prédictions
            output_path: Chemin de sortie (optionnel)
            
        Returns:
            Chemin du fichier sauvegardé
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"../data/predictions/predictions_{timestamp}.csv"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        predictions.to_csv(output_path, index=False)
        print(f"✅ Prédictions exportées vers: {output_path}")
        
        return output_path
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retourne les informations sur les modèles chargés
        
        Returns:
            Dictionnaire des informations
        """
        return {
            "models_loaded": self.is_loaded,
            "available_models": [name for name, model in self.models.items() if model is not None],
            "feature_count": len(self.models.get('feature_names', [])),
            "features_sample": self.models.get('feature_names', [])[:10],
            "model_info": self.model_info
        }
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Retourne l'importance des features du modèle
        
        Args:
            top_n: Nombre de features à retourner
            
        Returns:
            DataFrame avec l'importance des features
        """
        classifier = self.models.get('classifier')
        feature_names = self.models.get('feature_names')
        
        if classifier is None or feature_names is None:
            return pd.DataFrame()
        
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
        elif hasattr(classifier, 'coef_'):
            importances = np.abs(classifier.coef_[0])
        else:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df


# ============================================================
# FONCTIONS DE COMPATIBILITE
# ============================================================

# Instance globale pour réutilisation (singleton)
_prediction_service = None

def get_prediction_service() -> PredictionService:
    """Retourne l'instance singleton du service de prédiction"""
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service


def load_model(filename: str, models_path: str = '../models') -> Any:
    """Charge un modèle joblib"""
    try:
        full_path = os.path.join(models_path, filename)
        return joblib.load(full_path)
    except Exception as e:
        print(f"Erreur chargement {filename}: {e}")
        return None


def batch_predict_from_dict(customers_list: List[Dict], models: Dict = None) -> pd.DataFrame:
    """Compatibilité avec l'ancienne interface"""
    service = get_prediction_service()
    if not service.is_loaded:
        service.load_all_models()
    return service.predict_batch(customers_list)


# ============================================================
# UTILITAIRES POUR LES TESTS
# ============================================================

def create_sample_customer() -> Dict:
    """Crée un client exemple avec des valeurs par défaut"""
    sample = {
        # Features numériques de base
        'Recency': 50,
        'Frequency': 5,
        'MonetaryTotal': 1000.0,
        'MonetaryAvg': 200.0,
        'MonetaryStd': 50.0,
        'MonetaryMin': 10.0,
        'MonetaryMax': 500.0,
        'TotalQuantity': 100,
        'AvgQuantityPerTransaction': 10.0,
        'MinQuantity': 1,
        'MaxQuantity': 50,
        'CustomerTenureDays': 200,
        'FirstPurchaseDaysAgo': 180,
        'PreferredDayOfWeek': 3,
        'PreferredHour': 12,
        'PreferredMonth': 6,
        'WeekendPurchaseRatio': 0.3,
        'AvgDaysBetweenPurchases': 30.0,
        'UniqueProducts': 20,
        'UniqueDescriptions': 20,
        'AvgProductsPerTransaction': 5.0,
        'UniqueCountries': 1,
        'NegativeQuantityCount': 0,
        'ZeroPriceCount': 0,
        'CancelledTransactions': 0,
        'ReturnRatio': 0.05,
        'TotalTransactions': 5,
        'UniqueInvoices': 5,
        'AvgLinesPerInvoice': 10.0,
        'Age': 35,
        'SupportTicketsCount': 1,
        'SatisfactionScore': 4,
        'IPClass_A': 1,
        'IPClass_B': 0,
        'IPClass_C': 0,
        
        # Variables ordinales encodées
        'AgeCategory_encoded': 2,
        'SpendingCategory_encoded': 2,
        'LoyaltyLevel_encoded': 2,
        'ChurnRiskCategory_encoded': 2,
        'BasketSizeCategory_encoded': 1,
        'PreferredTimeOfDay_encoded': 1,
        'RFMSegment_encoded': 2,
        
        # One-hot encodings CustomerType
        'CustomerType_Hyperactif': 0,
        'CustomerType_Nouveau': 0,
        'CustomerType_Occasionnel': 0,
        'CustomerType_Perdu': 0,
        'CustomerType_Régulier': 1,
        
        # One-hot encodings FavoriteSeason
        'FavoriteSeason_Automne': 0,
        'FavoriteSeason_Été': 0,
        'FavoriteSeason_Hiver': 1,
        'FavoriteSeason_Printemps': 0,
        
        # One-hot encodings WeekendPreference
        'WeekendPreference_Inconnu': 0,
        'WeekendPreference_Semaine': 1,
        'WeekendPreference_Weekend': 0,
        
        # One-hot encodings ProductDiversity
        'ProductDiversity_Explorateur': 1,
        'ProductDiversity_Modéré': 0,
        'ProductDiversity_Spécialisé': 0,
        
        # One-hot encodings Gender
        'Gender_F': 0,
        'Gender_M': 1,
        'Gender_Unknown': 0,
        
        # One-hot encodings AccountStatus
        'AccountStatus_Active': 1,
        'AccountStatus_Closed': 0,
        'AccountStatus_Pending': 0,
        'AccountStatus_Suspended': 0,
        
        # Country encodings (exemple pour UK)
        'Country_United Kingdom': 1,
    }
    return sample


def create_risk_profiles() -> List[Dict]:
    """Crée plusieurs profils de clients avec différents niveaux de risque"""
    base = create_sample_customer()
    
    profiles = [
        {
            'name': 'Client à faible risque',
            'data': {**base, 'Frequency': 20, 'MonetaryTotal': 5000, 'Recency': 10,
                     'SupportTicketsCount': 0, 'ReturnRatio': 0.01}
        },
        {
            'name': 'Client à risque modéré',
            'data': {**base, 'Frequency': 5, 'MonetaryTotal': 1000, 'Recency': 50,
                     'SupportTicketsCount': 1, 'ReturnRatio': 0.05}
        },
        {
            'name': 'Client à risque élevé',
            'data': {**base, 'Frequency': 2, 'MonetaryTotal': 200, 'Recency': 150,
                     'SupportTicketsCount': 3, 'ReturnRatio': 0.15}
        },
        {
            'name': 'Client à risque critique',
            'data': {**base, 'Frequency': 1, 'MonetaryTotal': 50, 'Recency': 300,
                     'SupportTicketsCount': 5, 'ReturnRatio': 0.3}
        }
    ]
    
    return profiles


# ============================================================
# FONCTION PRINCIPALE DE TEST
# ============================================================

def main():
    """Test du service de prédiction"""
    print("\n" + "="*70)
    print(" " * 20 + "TEST DU SERVICE DE PREDICTION")
    print("="*70)
    
    service = get_prediction_service()
    models = service.load_all_models()
    
    if not service.is_loaded:
        print("\n❌ Aucun modèle trouvé. Exécutez d'abord train_model.py")
        print("\nVérifiez que les fichiers suivants existent dans ../models/:")
        print("  - best_model.pkl")
        print("  - scaler.pkl")
        print("  - feature_names.pkl")
        return
    
    # Informations des modèles
    print("\n" + "="*70)
    print(" " * 25 + "INFORMATIONS DES MODELES")
    print("="*70)
    info = service.get_model_info()
    print(f"  Modèles chargés: {info['models_loaded']}")
    print(f"  Nombre de features: {info['feature_count']}")
    print(f"  Modèles disponibles: {info['available_models']}")
    
    # Feature importance
    print("\n" + "="*70)
    print(" " * 22 + "TOP 10 FEATURES IMPORTANTES")
    print("="*70)
    importance_df = service.get_feature_importance(10)
    if not importance_df.empty:
        for idx, row in importance_df.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    else:
        print("  Non disponible pour ce modèle")
    
    # Test avec différents profils
    print("\n" + "="*70)
    print(" " * 22 + "PREDICTION PAR PROFIL DE RISQUE")
    print("="*70)
    
    profiles = create_risk_profiles()
    
    results_list = []
    for profile in profiles:
        print(f"\n--- {profile['name']} ---")
        result = service.predict_single_customer(profile['data'])
        
        if result.get('success'):
            print(f"  Churn prédit: {'🔴 OUI' if result['churn_prediction'] == 1 else '🟢 NON'}")
            print(f"  Probabilité: {result['churn_probability']:.2%}")
            print(f"  Risque: {result['churn_risk']}")
            print(f"  Recommandation: {result['churn_recommendation']}")
            
            results_list.append({
                'profile': profile['name'],
                'churn_prediction': result['churn_prediction'],
                'churn_probability': result['churn_probability'],
                'risk_level': result['churn_risk_level']
            })
        else:
            print(f"  ❌ Erreur: {result.get('error', 'Inconnue')}")
    
    # Tableau récapitulatif
    print("\n" + "="*70)
    print(" " * 25 + "RECAPITULATIF DES PREDICTIONS")
    print("="*70)
    results_df = pd.DataFrame(results_list)
    print(results_df.to_string(index=False))
    
    # Test batch
    print("\n" + "="*70)
    print(" " * 28 + "TEST BATCH (3 clients)")
    print("="*70)
    
    customers_data = [p['data'] for p in profiles[:3]]
    batch_results = service.predict_batch(customers_data, include_details=True)
    print(batch_results.to_string(index=False))
    
    # Export des résultats
    print("\n" + "="*70)
    print(" " * 28 + "EXPORT DES RESULTATS")
    print("="*70)
    
    if len(results_list) > 0:
        export_df = pd.DataFrame(results_list)
        output_path = service.export_predictions(export_df)
    
    print("\n" + "="*70)
    print(" " * 25 + "TEST TERMINE AVEC SUCCES")
    print("="*70)
    
    return service


# ============================================================
# POINT D'ENTREE
# ============================================================

if __name__ == "__main__":
    service = main()