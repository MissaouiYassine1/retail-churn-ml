"""
UTILITAIRES POUR LA VISUALISATION ET LA PREDICTION
===================================================
Fonctions pour visualiser les résultats des modèles et faire des prédictions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os
import joblib
from typing import Dict, List, Any, Tuple, Optional

# Créer le dossier figures s'il n'existe pas
FIGURE_PATH = '../figures'
os.makedirs(FIGURE_PATH, exist_ok=True)

# ============================================================
# FONCTIONS DE CHARGEMENT DE MODELES
# ============================================================

def load_model(filename: str, models_path: str = '../models') -> Any:
    """
    Charge un modèle joblib
    
    Args:
        filename: Nom du fichier (ex: 'classifier.joblib')
        models_path: Chemin vers le dossier des modèles
    
    Returns:
        Modèle chargé ou None si erreur
    """
    try:
        full_path = os.path.join(models_path, filename)
        model = joblib.load(full_path)
        print(f"✅ Modèle chargé: {filename}")
        return model
    except FileNotFoundError:
        print(f"⚠️ Fichier non trouvé: {full_path}")
        return None
    except Exception as e:
        print(f"❌ Erreur chargement {filename}: {e}")
        return None

def save_model(model: Any, filename: str, models_path: str = '../models') -> bool:
    """
    Sauvegarde un modèle
    
    Args:
        model: Modèle à sauvegarder
        filename: Nom du fichier
        models_path: Chemin vers le dossier des modèles
    
    Returns:
        True si succès, False sinon
    """
    try:
        os.makedirs(models_path, exist_ok=True)
        full_path = os.path.join(models_path, filename)
        joblib.dump(model, full_path)
        print(f"✅ Modèle sauvegardé: {filename}")
        return True
    except Exception as e:
        print(f"❌ Erreur sauvegarde {filename}: {e}")
        return False

def load_all_models(model_names: List[str] = None) -> Dict[str, Any]:
    """
    Charge plusieurs modèles
    
    Args:
        model_names: Liste des noms de modèles à charger
    
    Returns:
        Dictionnaire des modèles chargés
    """
    if model_names is None:
        model_names = [
            'scaler', 'feature_names', 'pca', 'kmeans', 
            'classifier', 'regressor', 'label_encoder_country'
        ]
    
    models = {}
    for model_name in model_names:
        models[model_name] = load_model(f'{model_name}.joblib')
    
    return models

# ============================================================
# FONCTIONS DE PREDICTION BATCH
# ============================================================

def batch_predict_from_dict(customers_list: List[Dict], models: Dict[str, Any]) -> pd.DataFrame:
    """
    Prédiction batch à partir d'une liste de dictionnaires
    
    Args:
        customers_list: Liste de dictionnaires contenant les features des clients
        models: Dictionnaire des modèles chargés (doit contenir 'classifier', 'kmeans', etc.)
    
    Returns:
        DataFrame avec les prédictions
    """
    if len(customers_list) == 0:
        return pd.DataFrame()
    
    # Créer DataFrame
    df = pd.DataFrame(customers_list)
    
    # Aligner les features
    feature_names = models.get('feature_names')
    if feature_names is not None:
        for feat in feature_names:
            if feat not in df.columns:
                df[feat] = 0
        df = df[feature_names]
    
    X = df.values
    
    # Appliquer scaler si disponible
    scaler = models.get('scaler')
    if scaler is not None:
        X = scaler.transform(X)
    
    results = pd.DataFrame()
    
    # Prédictions churn
    classifier = models.get('classifier')
    if classifier is not None:
        results['churn_prediction'] = classifier.predict(X)
        try:
            results['churn_probability'] = classifier.predict_proba(X)[:, 1]
        except:
            results['churn_probability'] = results['churn_prediction'].astype(float)
    
    # Prédictions cluster
    kmeans = models.get('kmeans')
    if kmeans is not None:
        results['cluster'] = kmeans.predict(X)
    
    # Prédictions monetary
    regressor = models.get('regressor')
    if regressor is not None:
        results['predicted_monetary'] = regressor.predict(X)
    
    return results

def batch_predict_from_df(df: pd.DataFrame, models: Dict[str, Any]) -> pd.DataFrame:
    """
    Prédiction batch à partir d'un DataFrame
    
    Args:
        df: DataFrame contenant les features des clients
        models: Dictionnaire des modèles chargés
    
    Returns:
        DataFrame avec les prédictions
    """
    customers_list = df.to_dict(orient='records')
    return batch_predict_from_dict(customers_list, models)

# ============================================================
# FONCTIONS DE CHARGEMENT DES DONNEES
# ============================================================

def load_train_test_data(data_path: str = '../data/train_test') -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Charge les données train/test
    
    Args:
        data_path: Chemin vers le dossier des données
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train = pd.read_csv(f'{data_path}/X_train.csv')
    X_test = pd.read_csv(f'{data_path}/X_test.csv')
    y_train = pd.read_csv(f'{data_path}/y_train.csv').values.ravel()
    y_test = pd.read_csv(f'{data_path}/y_test.csv').values.ravel()
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train distribution:\n{pd.Series(y_train).value_counts()}")
    
    return X_train, X_test, y_train, y_test

def load_processed_data(data_path: str = '../data/processed') -> pd.DataFrame:
    """
    Charge les données traitées complètes
    """
    df = pd.read_csv(f'{data_path}/retail_customers_processed.csv')
    print(f"Data shape: {df.shape}")
    return df

# ============================================================
# FONCTIONS DE VISUALISATION
# ============================================================

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Affiche et sauvegarde la matrice de confusion
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Sauvegarder
    filename = title.replace(' ', '_').replace('-', '_')
    plt.savefig(f'{FIGURE_PATH}/{filename}.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    # Calculer les métriques
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nMatrice de confusion - {title}:")
    print(f"  True Negatives: {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives: {tp}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    return cm

def plot_roc_curve(y_true, y_pred_proba, title="ROC Curve"):
    """
    Affiche et sauvegarde la courbe ROC
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Sauvegarder
    filename = title.replace(' ', '_').replace('-', '_')
    plt.savefig(f'{FIGURE_PATH}/{filename}.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    return fpr, tpr, roc_auc

def plot_feature_importance(importances, feature_names, title="Feature Importance", top_n=20):
    """
    Affiche et sauvegarde l'importance des features
    """
    # Créer un DataFrame avec les importances
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Prendre les top_n features
    importance_df = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['feature'], importance_df['importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Sauvegarder
    filename = title.replace(' ', '_').replace('-', '_')
    plt.savefig(f'{FIGURE_PATH}/{filename}.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    return importance_df

def plot_pca_2d(X_pca, labels, title="ACP Visualization"):
    """
    Visualisation 2D de l'ACP
    Gère à la fois les cas avec 1 et 2 composantes principales
    """
    plt.figure(figsize=(10, 8))
    
    # Vérifier le nombre de dimensions
    if X_pca.shape[1] == 1:
        # Si une seule composante, créer un scatter plot 1D
        y_coords = np.zeros_like(X_pca[:, 0])
        scatter = plt.scatter(X_pca[:, 0], y_coords, c=labels, cmap='viridis', alpha=0.6)
        plt.xlabel('Composante Principale 1')
        plt.ylabel('(Seulement une dimension)')
        plt.title(f"{title} (1 composante)")
        print("Note: Visualisation 1D car ACP n'a qu'une seule composante")
    else:
        # Visualisation 2D normale
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.xlabel('Composante Principale 1')
        plt.ylabel('Composante Principale 2')
        plt.title(title)
    
    plt.colorbar(scatter, label='Churn')
    plt.grid(True, alpha=0.3)
    
    # Sauvegarder
    filename = title.replace(' ', '_').replace('-', '_')
    plt.savefig(f'{FIGURE_PATH}/{filename}.png', dpi=100, bbox_inches='tight')
    plt.show()

def plot_learning_curve(train_scores, val_scores, title="Learning Curve"):
    """Affiche la courbe d'apprentissage"""
    plt.figure(figsize=(8, 6))
    plt.plot(train_scores, label='Train Score', marker='o')
    plt.plot(val_scores, label='Validation Score', marker='o')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = title.replace(' ', '_').replace('-', '_')
    plt.savefig(f'{FIGURE_PATH}/{filename}.png', dpi=100, bbox_inches='tight')
    plt.show()

def plot_precision_recall_curve(y_true, y_pred_proba, title="Precision-Recall Curve"):
    """Affiche la courbe Precision-Recall"""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    ap_score = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, 
             label=f'PR curve (AP = {ap_score:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    filename = title.replace(' ', '_').replace('-', '_')
    plt.savefig(f'{FIGURE_PATH}/{filename}.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    return precision, recall, ap_score

def plot_correlation_heatmap(df, title="Correlation Heatmap", figsize=(12, 10)):
    """Affiche la heatmap des corrélations"""
    plt.figure(figsize=figsize)
    
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(title)
    
    filename = title.replace(' ', '_').replace('-', '_')
    plt.savefig(f'{FIGURE_PATH}/{filename}.png', dpi=100, bbox_inches='tight')
    plt.show()

def plot_class_distribution(y_train, y_test, title="Class Distribution"):
    """Affiche la distribution des classes"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Distribution train
    train_counts = pd.Series(y_train).value_counts()
    ax1.bar(['No Churn', 'Churn'], train_counts.values, color=['green', 'red'])
    ax1.set_title('Train Set')
    ax1.set_xlabel('Churn')
    ax1.set_ylabel('Count')
    
    for i, (label, count) in enumerate(train_counts.items()):
        pct = count / len(y_train) * 100
        ax1.text(i, count + 5, f'{pct:.1f}%', ha='center')
    
    # Distribution test
    test_counts = pd.Series(y_test).value_counts()
    ax2.bar(['No Churn', 'Churn'], test_counts.values, color=['green', 'red'])
    ax2.set_title('Test Set')
    ax2.set_xlabel('Churn')
    ax2.set_ylabel('Count')
    
    for i, (label, count) in enumerate(test_counts.items()):
        pct = count / len(y_test) * 100
        ax2.text(i, count + 1, f'{pct:.1f}%', ha='center')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    filename = title.replace(' ', '_').replace('-', '_')
    plt.savefig(f'{FIGURE_PATH}/{filename}.png', dpi=100, bbox_inches='tight')
    plt.show()