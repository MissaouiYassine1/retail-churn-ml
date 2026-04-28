# app/app_api.py - Version qui communique avec l'API
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# Configuration API
API_URL = "http://localhost:8000"  # URL de votre API

# Titre
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("---")

# Vérifier la connexion à l'API
@st.cache_resource
def check_api_connection():
    """Vérifie si l'API est accessible"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return True, data
        return False, None
    except Exception as e:
        return False, str(e)

api_connected, api_info = check_api_connection()

if not api_connected:
    st.error(f"❌ Impossible de se connecter à l'API sur {API_URL}")
    st.info("Assurez-vous que l'API est lancée avec: py -3.10 app/api.py")
    st.stop()

st.success(f"✅ Connecté à l'API - {api_info.get('feature_count', 0)} features")

# Sidebar
st.sidebar.header("🎯 Navigation")
page = st.sidebar.radio("Choisir une page", [
    "📝 Prédiction Simple",
    "📊 Batch Prediction",
    "📈 Analyse des Risques",
    "ℹ️ Informations"
])

# Fonction pour appeler l'API
def call_prediction_api(customer_data):
    """Appelle l'API de prédiction"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=customer_data,
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Erreur API: {response.status_code}"
    except Exception as e:
        return None, str(e)

def call_batch_api(customers_list):
    """Appelle l'API batch"""
    try:
        response = requests.post(
            f"{API_URL}/predict/batch",
            json=customers_list,
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Erreur API: {response.status_code}"
    except Exception as e:
        return None, str(e)

# Page 1: Prédiction Simple
if page == "📝 Prédiction Simple":
    st.header("🎯 Prédiction de Churn - Client Unique")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Informations Client")
        recency = st.number_input("Recency (jours depuis dernier achat)", min_value=0, value=50)
        frequency = st.number_input("Frequency (nombre d'achats)", min_value=1, value=5)
        monetary = st.number_input("MonetaryTotal (dépenses totales)", min_value=0.0, value=1000.0)
        quantity = st.number_input("TotalQuantity (produits achetés)", min_value=0, value=100)
        
    with col2:
        st.subheader("Comportement Client")
        age = st.number_input("Âge", min_value=18, max_value=100, value=35)
        tickets = st.number_input("Support Tickets", min_value=0, value=1)
        satisfaction = st.slider("Satisfaction Score", 1, 5, 4)
        returns = st.slider("Return Ratio", 0.0, 1.0, 0.05, 0.01)
    
    if st.button("🔮 Prédire le Churn", type="primary"):
        customer_data = {
            'Recency': recency,
            'Frequency': frequency,
            'MonetaryTotal': monetary,
            'TotalQuantity': quantity,
            'Age': age,
            'SupportTicketsCount': tickets,
            'SatisfactionScore': satisfaction,
            'ReturnRatio': returns
        }
        
        with st.spinner("Prédiction en cours..."):
            result, error = call_prediction_api(customer_data)
        
        if result:
            st.markdown("---")
            st.subheader("📊 Résultats")
            
            col1, col2, col3 = st.columns(3)
            
            color = "🔴" if result['churn_prediction'] == 1 else "🟢"
            
            with col1:
                st.metric("Prédiction Churn", f"{color} {'OUI' if result['churn_prediction'] == 1 else 'NON'}")
            
            with col2:
                st.metric("Probabilité", f"{result['churn_probability']:.2%}")
            
            with col3:
                st.metric("Niveau de Risque", result['churn_risk_level'])
            
            st.info(f"📝 {result['churn_risk_description']}")
            st.warning(f"💡 Recommandation: {result['churn_recommendation']}")
            
            # Jauge de risque
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['churn_probability'] * 100,
                title={'text': "Risque de Churn (%)"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred" if result['churn_probability'] > 0.5 else "darkgreen"},
                    'steps': [
                        {'range': [0, 20], 'color': "lightgreen"},
                        {'range': [20, 40], 'color': "yellowgreen"},
                        {'range': [40, 60], 'color': "orange"},
                        {'range': [60, 80], 'color': "salmon"},
                        {'range': [80, 100], 'color': "red"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Erreur: {error}")

# Page 2: Batch Prediction
elif page == "📊 Batch Prediction":
    st.header("📊 Prédiction Batch via API")
    
    st.markdown("### Upload de fichier CSV")
    st.markdown("Le fichier doit contenir les colonnes: Recency, Frequency, MonetaryTotal, TotalQuantity, Age")
    
    uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Aperçu des données:", df.head())
        
        if st.button("🚀 Lancer les prédictions"):
            with st.spinner("Prédictions en cours..."):
                # Convertir DataFrame en liste de dictionnaires
                customers_list = df.to_dict(orient='records')
                result, error = call_batch_api(customers_list)
            
            if result:
                st.success("✅ Prédictions terminées!")
                
                # Créer DataFrame des résultats
                results_df = pd.DataFrame(result['results'])
                st.write("Résultats:", results_df.head())
                
                # Statistiques
                st.markdown("### 📈 Statistiques des prédictions")
                col1, col2, col3 = st.columns(3)
                
                if 'churn_prediction' in results_df.columns:
                    churn_rate = results_df['churn_prediction'].mean()
                    avg_prob = results_df['churn_probability'].mean() if 'churn_probability' in results_df.columns else 0
                    high_risk = len(results_df[results_df['churn_risk_level'].isin(['High', 'Critical'])]) if 'churn_risk_level' in results_df.columns else 0
                    
                    with col1:
                        st.metric("Taux de Churn Prédit", f"{churn_rate:.1%}")
                    with col2:
                        st.metric("Probabilité Moyenne", f"{avg_prob:.1%}")
                    with col3:
                        st.metric("Clients Haut Risque", high_risk)
                
                # Download
                csv = results_df.to_csv(index=False)
                st.download_button("📥 Télécharger les résultats", csv, "predictions.csv", "text/csv")
            else:
                st.error(f"Erreur: {error}")

# Page 3: Informations API
elif page == "ℹ️ Informations":
    st.header("ℹ️ Informations sur l'API")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Statut de l'API")
        st.write(f"URL: {API_URL}")
        st.write(f"Connecté: ✅")
        st.write(f"Features: {api_info.get('feature_count', 'N/A')}")
    
    with col2:
        st.markdown("### Endpoints disponibles")
        st.write("- GET `/health` - Vérification santé")
        st.write("- GET `/model/info` - Infos modèles")
        st.write("- POST `/predict` - Prédiction simple")
        st.write("- POST `/predict/batch` - Prédiction batch")
    
    st.markdown("### 📊 Documentation API")
    st.markdown(f"Consultez la documentation interactive: [Swagger UI]({API_URL}/docs)")

st.markdown("---")
st.caption("Churn Prediction Dashboard - Communique avec l'API FastAPI")