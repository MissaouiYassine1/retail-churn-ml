import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import os

def load_data(filepath):
    """Charge les données brutes"""
    print("============================================================")
    print("CHARGEMENT DES DONNEES")
    print("============================================================")
    df = pd.read_csv(filepath)
    print(f"Donnees chargees: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df

def explore_data(df):
    """Analyse exploratoire basique"""
    print("============================================================")
    print("EXPLORATION DES DONNEES")
    print("============================================================")
    print(f"\nDimensions: {df.shape[0]} lignes x {df.shape[1]} colonnes")
    
    print("\n--- Types de donnees ---")
    print(df.dtypes.value_counts())
    
    print("\n--- Valeurs manquantes ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing': missing, 'Percentage': missing_pct})
    missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("Aucune valeur manquante detectee")
    
    print("\n--- Statistiques descriptives (numeriques) ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numeric_cols].describe())
    
    # Statistiques sur Churn
    if 'Churn' in df.columns:
        print(f"\nDistribution de Churn: {df['Churn'].value_counts().to_dict()}")
        print(f"Taux de churn: {df['Churn'].mean():.2%}")

def engineer_features(df):
    """Création de nouvelles features"""
    df_feat = df.copy()
    
    # 1. Ratio de la valeur moyenne par transaction
    df_feat['AvgValuePerTransaction'] = df_feat['MonetaryAvg'] / df_feat['AvgQuantityPerTransaction'].replace(0, np.nan)
    
    # 2. Coefficient de variation des prix
    df_feat['MonetaryCV'] = df_feat['MonetaryStd'] / (df_feat['MonetaryAvg'].abs() + 1e-6)
    
    # 3. Ratio retour/achat
    df_feat['ReturnRatioSquared'] = df_feat['ReturnRatio'] ** 2
    
    # 4. Profondeur du panier
    df_feat['BasketDepth'] = df_feat['AvgLinesPerInvoice'] / df_feat['AvgQuantityPerTransaction'].replace(0, np.nan)
    
    # 5. Taux d'annulation
    if 'CancelledTransactions' in df_feat.columns:
        df_feat['CancellationRate'] = df_feat['CancelledTransactions'] / (df_feat['TotalTransactions'] + 1)
    
    return df_feat

def safe_label_encode(df, col, categories):
    """Label encoding sécurisé qui gère les valeurs inconnues"""
    df[col] = df[col].fillna('Unknown')
    df[col] = df[col].replace('Inconn', 'Inconnu')
    
    extended_categories = categories.copy()
    if 'Unknown' not in extended_categories:
        extended_categories.append('Unknown')
    
    mapping = {cat: idx for idx, cat in enumerate(extended_categories)}
    df[col] = df[col].apply(lambda x: x if x in mapping else 'Unknown')
    df[col] = df[col].map(mapping)
    
    return df

def encode_all_categorical_columns(X):
    """Encode TOUTES les colonnes catégorielles restantes"""
    print("\n--- Encodage des colonnes catégorielles residuelles ---")
    
    # Identifier les colonnes de type object (texte)
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Colonnes categorielle a encoder: {categorical_cols}")
    
    # One-Hot encoder pour toutes les colonnes restantes
    if categorical_cols:
        # Utiliser OneHotEncoder pour les colonnes restantes
        X_encoded = pd.get_dummies(X, columns=categorical_cols, prefix=categorical_cols)
        print(f"One-Hot encode des colonnes restantes: {len(categorical_cols)} colonnes -> {X_encoded.shape[1]} colonnes")
        return X_encoded
    else:
        print("Aucune colonne categorielle restante a encoder")
        return X

def load_and_preprocess(df, ordinal_features):
    """Applique le preprocessing aux données"""
    
    # 1. Séparer X et y
    if 'Churn' in df.columns:
        X = df.drop('Churn', axis=1)
        y = df['Churn']
    else:
        X = df
        y = None
    
    # 2. Détection des outliers
    for col in ['MonetaryTotal', 'TotalQuantity', 'MonetaryMin', 'MonetaryMax']:
        if col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((X[col] < (Q1 - 1.5 * IQR)) | (X[col] > (Q3 + 1.5 * IQR))).sum()
            print(f"{col}: {outliers} outliers detectes (IQR method)")
    
    # 3. Engineering features
    X = engineer_features(X)
    print("Feature engineering: nouvelles features creees")
    
    # 4. Encodage ordinal
    for col, categories in ordinal_features.items():
        if col in X.columns:
            X = safe_label_encode(X, col, categories)
            print(f"Encoded ordinal: {col}")
    
    # 5. One-Hot Encoding pour colonnes spécifiques
    nominal_columns = ['CustomerType', 'FavoriteSeason', 'Region', 'WeekendPreference', 
                       'ProductDiversity', 'Gender', 'AccountStatus']
    for col in nominal_columns:
        if col in X.columns:
            X[col] = X[col].fillna('Unknown')
            X[col] = X[col].replace('Inconn', 'Inconnu')
            dummies = pd.get_dummies(X[col], prefix=col)
            X = pd.concat([X, dummies], axis=1)
            X = X.drop(col, axis=1)
            print(f"One-Hot encoded: {col} -> {dummies.shape[1]} colonnes")
    
    # 6. Traitement des IP
    if 'LastLoginIP' in X.columns:
        X['LastLoginIP'] = X['LastLoginIP'].fillna('Unknown')
        X['IPClass'] = X['LastLoginIP'].apply(
            lambda x: 'Class_A' if str(x).startswith(('10.', '172.16.', '192.168.')) else
                      'Class_B' if str(x).split('.')[0].isdigit() and 1 <= int(str(x).split('.')[0]) <= 126 else
                      'Class_C' if str(x) != 'Unknown' else
                      'Unknown'
        )
        dummies_ip = pd.get_dummies(X['IPClass'], prefix='IPClass')
        X = pd.concat([X, dummies_ip], axis=1)
        X = X.drop(['LastLoginIP', 'IPClass'], axis=1)
        print(f"Encoded IPClass -> {dummies_ip.shape[1]} colonnes")
    
    # 7. Supprimer les colonnes inutiles
    cols_to_drop = ['RegistrationDate', 'NewsletterSubscribed', 'CustomerID']
    for col in cols_to_drop:
        if col in X.columns:
            X = X.drop(col, axis=1)
            print(f"Suppression de {col}")
    
    # 8. ⚠️ CRUCIAL: Encoder TOUTES les autres colonnes catégorielles restantes
    X = encode_all_categorical_columns(X)
    
    # 9. Imputation numérique
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='median')
    X[numeric_cols] = imputer.fit_transform(X[numeric_cols])
    print(f"Imputation numerique (median): {len(numeric_cols)} colonnes")
    
    # 10. Vérification finale - plus aucune colonne de type object ne devrait rester
    remaining_object_cols = X.select_dtypes(include=['object']).columns.tolist()
    if remaining_object_cols:
        print(f"⚠️ ATTENTION: Colonnes object restantes: {remaining_object_cols}")
        # Forcer l'encodage de ces colonnes
        for col in remaining_object_cols:
            print(f"  - Encodage forcé de {col}")
            dummies = pd.get_dummies(X[col], prefix=col)
            X = pd.concat([X, dummies], axis=1)
            X = X.drop(col, axis=1)
        print(f"After forced encoding: {X.shape}")
    else:
        print("✅ Toutes les colonnes sont numeriques!")
    
    return X, y

def main():
    # Configuration
    input_path = r"C:\Users\Lenovo\Desktop\projet_ml_retail1\data\raw\retail_customers_COMPLETE_CATEGORICAL.csv"
    output_dir = r"C:\Users\Lenovo\Desktop\projet_ml_retail1\data\processed"
    model_dir = r"C:\Users\Lenovo\Desktop\projet_ml_retail1\models"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Chargement
    df = load_data(input_path)
    
    # 2. Exploration
    explore_data(df)
    
    # 3. Définir les colonnes ordinales
    ordinal_features = {
        'AgeCategory': ['Inconnu', '18-24', '25-34', '35-44', '45-54', '55-64', '65+'],
        'SpendingCategory': ['Low', 'Medium', 'High', 'VIP'],
        'LoyaltyLevel': ['Jeune', 'Établi', 'Ancien'],
        'ChurnRiskCategory': ['Faible', 'Moyen', 'Élevé', 'Critique'],
        'BasketSizeCategory': ['Petit', 'Moyen', 'Grand'],
        'PreferredTimeOfDay': ['Matin', 'Midi', 'Après-midi', 'Soir'],
        'RFMSegment': ['Dormants', 'Potentiels', 'Fidèles', 'Champions']
    }
    
    # 4. Appliquer le preprocessing
    X_processed, y_processed = load_and_preprocess(df, ordinal_features)
    
    # 5. Sauvegarder
    output_file = os.path.join(output_dir, "retail_customers_processed.csv")
    df_processed = X_processed.copy()
    if y_processed is not None:
        df_processed['Churn'] = y_processed
    df_processed.to_csv(output_file, index=False)
    print(f"\n✅ Donnees sauvegardees: {output_file}")
    print(f"   Shape: {df_processed.shape}")
    print(f"   Types: {df_processed.dtypes.value_counts()}")
    
    # 6. Créer les datasets train/test
    print("\n============================================================")
    print("CREATION DES DATASETS TRAIN/TEST")
    print("============================================================")
    
    if y_processed is not None:
        # Split des données
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=0.2, random_state=42, stratify=y_processed
        )
        
        # Sauvegarder les datasets
        train_test_dir = r"C:\Users\Lenovo\Desktop\projet_ml_retail1\data\train_test"
        os.makedirs(train_test_dir, exist_ok=True)
        
        X_train.to_csv(os.path.join(train_test_dir, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(train_test_dir, "X_test.csv"), index=False)
        pd.Series(y_train).to_csv(os.path.join(train_test_dir, "y_train.csv"), index=False, header=['Churn'])
        pd.Series(y_test).to_csv(os.path.join(train_test_dir, "y_test.csv"), index=False, header=['Churn'])
        
        print(f"✅ Train set: {X_train.shape[0]} lignes, {X_train.shape[1]} colonnes")
        print(f"✅ Test set: {X_test.shape[0]} lignes, {X_test.shape[1]} colonnes")
        print(f"   Taux de churn train: {y_train.mean():.2%}")
        print(f"   Taux de churn test: {y_test.mean():.2%}")
        
        # Vérifier qu'il n'y a plus de colonnes object
        object_cols_train = X_train.select_dtypes(include=['object']).columns.tolist()
        if object_cols_train:
            print(f"⚠️ ATTENTION: Colonnes object dans X_train: {object_cols_train}")
        else:
            print("✅ X_train ne contient que des colonnes numeriques")
        
        # Sauvegarder le scaler
        scaler = StandardScaler()
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        scaler.fit(X_processed[numeric_cols])
        joblib.dump(scaler, os.path.join(model_dir, "standard_scaler.joblib"))
        joblib.dump(numeric_cols.tolist(), os.path.join(model_dir, "feature_names.joblib"))
        print("\n✅ Modeles sauvegardes: scaler et feature names")
    else:
        print("❌ Colonne 'Churn' non trouvée. Impossible de créer les datasets train/test.")

if __name__ == "__main__":
    main()