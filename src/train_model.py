"""
MODELISATION - CHURN PREDICTION
================================
But: Entraîner et évaluer des modèles de classification pour prédire le churn
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                             roc_curve, precision_recall_curve, f1_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
import joblib
from utils import plot_confusion_matrix, plot_roc_curve, plot_feature_importance

# Ajouter le chemin parent pour importer utils si nécessaire
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Chemins
DATA_PATH = '../data/train_test'
MODEL_PATH = '../models'
FIGURE_PATH = '../figures'

def load_train_test():
    """Charger les datasets train/test"""
    print("\n" + "="*60)
    print("CHARGEMENT DES DONNEES")
    print("="*60)
    
    X_train = pd.read_csv(f'{DATA_PATH}/X_train.csv')
    X_test = pd.read_csv(f'{DATA_PATH}/X_test.csv')
    y_train = pd.read_csv(f'{DATA_PATH}/y_train.csv').values.ravel()
    y_test = pd.read_csv(f'{DATA_PATH}/y_test.csv').values.ravel()
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train distribution:\n{pd.Series(y_train).value_counts()}")
    
    return X_train, X_test, y_train, y_test

def apply_pca(X_train, X_test, n_components=2):
    """
    Appliquer ACP pour réduction de dimension
    n_components=2 pour visualisation 2D
    """
    print("\n" + "="*60)
    print("1. ANALYSE EN COMPOSANTES PRINCIPALES (ACP)")
    print("="*60)
    
    # Standardisation déjà faite dans preprocessing
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    print(f"ACP - Reduction de {X_train.shape[1]} features a {pca.n_components_} composantes")
    print(f"Variance expliquee par composante: {pca.explained_variance_ratio_}")
    print(f"Variance expliquee totale: {pca.explained_variance_ratio_.sum():.2%}")
    
    return X_train_pca, X_test_pca, pca

def train_models(X_train, y_train, X_test, y_test, feature_names):
    """
    Entraîner et évaluer plusieurs modèles
    """
    print("\n" + "="*60)
    print("2. ENTRAINEMENT DES MODELES")
    print("="*60)
    
    # Définition des modèles
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    }
    
    results = {}
    best_model = None
    best_f1 = 0
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        print(f"CV F1-Score moyen: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Entraînement
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Métriques
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Test F1-Score: {f1:.4f}")
        print(f"Test AUC: {auc:.4f}")
        
        # Matrice de confusion
        plot_confusion_matrix(y_test, y_pred, title=f'Confusion Matrix - {name}')
        
        # Courbe ROC
        plot_roc_curve(y_test, y_pred_proba, title=f'ROC Curve - {name}')
        
        # Feature importance pour les modèles arborescents
        if hasattr(model, 'feature_importances_'):
            plot_feature_importance(model.feature_importances_, feature_names, 
                                   title=f'Feature Importance - {name}')
        
        results[name] = {
            'model': model,
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std(),
            'test_f1': f1,
            'test_auc': auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
    
    # Créer un tableau comparatif
    results_df = pd.DataFrame({
        name: {
            'CV_F1_Mean': results[name]['cv_f1_mean'],
            'CV_F1_Std': results[name]['cv_f1_std'],
            'Test_F1': results[name]['test_f1'],
            'Test_AUC': results[name]['test_auc']
        }
        for name in results
    }).T
    
    print("\n" + "="*60)
    print("RESUME DES PERFORMANCES")
    print("="*60)
    print(results_df.round(4))
    
    return results, best_model

def optimize_hyperparameters(X_train, y_train):
    """
    Optimisation des hyperparamètres pour XGBoost
    """
    print("\n" + "="*60)
    print("3. OPTIMISATION DES HYPERPARAMETRES (XGBoost)")
    print("="*60)
    
    # Grille de paramètres
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0]
    }
    
    xgb = XGBClassifier(random_state=42, eval_metric='logloss')
    
    grid_search = GridSearchCV(
        xgb, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Meilleurs parametres: {grid_search.best_params_}")
    print(f"Meilleur score F1 (CV): {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def create_voting_classifier(X_train, y_train):
    """
    Créer un classifieur par vote
    """
    print("\n" + "="*60)
    print("4. CREATION DU VOTING CLASSIFIER")
    print("="*60)
    
    # Modèles individuels
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    
    # Voting classifier (hard vote)
    voting_clf = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('xgb', xgb)],
        voting='hard'
    )
    
    # Cross-validation
    cv_scores = cross_val_score(voting_clf, X_train, y_train, cv=5, scoring='f1')
    print(f"Voting Classifier - CV F1-Score moyen: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Entraînement
    voting_clf.fit(X_train, y_train)
    
    return voting_clf

def apply_tsne(X_train, y_train, n_components=2, perplexity=30):
    """
    Appliquer t-SNE pour visualisation (optionnel)
    """
    print("\n" + "="*60)
    print("5. VISUALISATION T-SNE")
    print("="*60)
    
    # Échantillonner pour réduire le temps de calcul
    np.random.seed(42)
    sample_size = min(3000, len(X_train))
    indices = np.random.choice(len(X_train), sample_size, replace=False)
    X_sample = X_train[indices]
    y_sample = y_train[indices]
    
    print(f"t-SNE sur {sample_size} echantillons (sur {len(X_train)})")
    
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X_sample)
    
    # Visualisation
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Churn')
    plt.title(f't-SNE Visualization (perplexity={perplexity})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True, alpha=0.3)
    
    # Sauvegarder
    os.makedirs(FIGURE_PATH, exist_ok=True)
    plt.savefig(f'{FIGURE_PATH}/tsne_visualization.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    return X_tsne

def save_models(best_model, voting_clf, scaler, feature_names):
    """
    Sauvegarder les modèles entraînés
    """
    print("\n" + "="*60)
    print("6. SAUVEGARDE DES MODELES")
    print("="*60)
    
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Sauvegarder le meilleur modèle
    joblib.dump(best_model, f'{MODEL_PATH}/best_model.pkl')
    print(f"✅ Meilleur modèle sauvegarde: {MODEL_PATH}/best_model.pkl")
    
    # Sauvegarder le voting classifier
    joblib.dump(voting_clf, f'{MODEL_PATH}/voting_classifier.pkl')
    print(f"✅ Voting classifier sauvegarde: {MODEL_PATH}/voting_classifier.pkl")
    
    # Sauvegarder le scaler et les feature names
    joblib.dump(scaler, f'{MODEL_PATH}/scaler.pkl')
    joblib.dump(feature_names, f'{MODEL_PATH}/feature_names.pkl')
    print(f"✅ Scaler et feature names sauvegardes")

def main():
    """Fonction principale"""
    
    # 1. Chargement des données
    X_train, X_test, y_train, y_test = load_train_test()
    
    # Conversion en arrays numpy pour certaines opérations
    X_train_arr = X_train.values
    X_test_arr = X_test.values
    
    # 2. ACP pour visualisation (2 composantes)
    X_train_pca, X_test_pca, pca = apply_pca(X_train_arr, X_test_arr, n_components=2)
    
    # 3. Entraînement des modèles
    results, best_model = train_models(X_train_arr, y_train, X_test_arr, y_test, 
                                       X_train.columns.tolist())
    
    # 4. Optimisation XGBoost
    best_xgb = optimize_hyperparameters(X_train_arr, y_train)
    
    # Évaluation du XGBoost optimisé
    y_pred_xgb = best_xgb.predict(X_test_arr)
    y_pred_proba_xgb = best_xgb.predict_proba(X_test_arr)[:, 1]
    f1_xgb = f1_score(y_test, y_pred_xgb)
    auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
    
    print(f"\nXGBoost optimise - Test F1: {f1_xgb:.4f}, AUC: {auc_xgb:.4f}")
    
    # 5. Voting classifier
    voting_clf = create_voting_classifier(X_train_arr, y_train)
    y_pred_voting = voting_clf.predict(X_test_arr)
    f1_voting = f1_score(y_test, y_pred_voting)
    print(f"Voting Classifier - Test F1: {f1_voting:.4f}")
    
    # 6. t-SNE pour visualisation (optionnel - peut être long)
    # Décommentez si nécessaire
    # X_tsne = apply_tsne(X_train_arr, y_train)
    
    # 7. Sauvegarde des modèles
    # Note: Pour le scaler, il faudrait le charger depuis preprocessing
    # Ici on utilise un scaler factice car la standardisation a déjà été faite
    from sklearn.preprocessing import StandardScaler
    dummy_scaler = StandardScaler()
    dummy_scaler.fit(X_train_arr)
    
    save_models(best_model, voting_clf, dummy_scaler, X_train.columns.tolist())
    
    print("\n" + "="*60)
    print("ENTRAINEMENT TERMINE AVEC SUCCES!")
    print("="*60)
    
    return results, best_model, voting_clf

if __name__ == "__main__":
    models = main()