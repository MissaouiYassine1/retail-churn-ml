# app/app.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.predict import get_prediction_service, create_sample_customer, create_risk_profiles

# Configuration de la page
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# Titre
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("---")

# Charger le service de prédiction
@st.cache_resource
def load_service():
    service = get_prediction_service()
    service.load_all_models()
    return service

try:
    service = load_service()
    if not service.is_loaded:
        st.error("❌ Modèles non chargés. Exécutez d'abord train_model.py")
        st.stop()
    st.success(f"✅ Modèles chargés - {service.model_info.get('feature_count', 0)} features")
except Exception as e:
    st.error(f"❌ Erreur chargement modèles: {e}")
    st.stop()

# Sidebar
st.sidebar.header("🎯 Navigation")
page = st.sidebar.radio("Choisir une page", [
    "📝 Prédiction Simple",
    "📊 Batch Prediction",
    "📈 Analyse des Risques",
    "ℹ️ Informations"
])

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
    
    # Bouton de prédiction
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
            result = service.predict_single_customer(customer_data)
        
        if result.get('success'):
            st.markdown("---")
            st.subheader("📊 Résultats")
            
            col1, col2, col3 = st.columns(3)
            
            # Couleur selon prédiction
            color = "🔴" if result['churn_prediction'] == 1 else "🟢"
            
            with col1:
                st.metric("Prédiction Churn", f"{color} {'OUI' if result['churn_prediction'] == 1 else 'NON'}")
            
            with col2:
                st.metric("Probabilité", f"{result['churn_probability']:.2%}")
            
            with col3:
                st.metric("Niveau de Risque", result['churn_risk_level'])
            
            st.info(f"📝 {result['churn_risk']}")
            st.warning(f"💡 Recommandation: {result['churn_recommendation']}")
            
            # Jauge de risque
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = result['churn_probability'] * 100,
                title = {'text': "Risque de Churn (%)"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred" if result['churn_probability'] > 0.5 else "darkgreen"},
                    'steps': [
                        {'range': [0, 20], 'color': "lightgreen"},
                        {'range': [20, 40], 'color': "yellowgreen"},
                        {'range': [40, 60], 'color': "orange"},
                        {'range': [60, 80], 'color': "salmon"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': result['churn_probability'] * 100
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Erreur: {result.get('error', 'Prédiction échouée')}")

# Page 2: Batch Prediction
elif page == "📊 Batch Prediction":
    st.header("📊 Prédiction Batch")
    
    st.markdown("### Upload de fichier CSV")
    st.markdown("Le fichier doit contenir les colonnes: Recency, Frequency, MonetaryTotal, TotalQuantity, Age")
    
    uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Aperçu des données:", df.head())
        
        if st.button("🚀 Lancer les prédictions"):
            with st.spinner("Prédictions en cours..."):
                result_df = service.predict_from_dataframe(df)
            
            st.success("✅ Prédictions terminées!")
            st.write("Résultats:", result_df.head())
            
            # Statistiques
            st.markdown("### 📈 Statistiques des prédictions")
            col1, col2, col3 = st.columns(3)
            
            churn_rate = result_df['churn_prediction'].mean()
            avg_prob = result_df['churn_probability'].mean()
            high_risk = (result_df['churn_risk_level'] == 'High').sum() + (result_df['churn_risk_level'] == 'Critical').sum()
            
            with col1:
                st.metric("Taux de Churn Prédit", f"{churn_rate:.1%}")
            with col2:
                st.metric("Probabilité Moyenne", f"{avg_prob:.1%}")
            with col3:
                st.metric("Clients Haut Risque", high_risk)
            
            # Distribution des risques
            risk_counts = result_df['churn_risk_level'].value_counts()
            fig = px.bar(x=risk_counts.index, y=risk_counts.values, 
                        title="Distribution des Niveaux de Risque",
                        color=risk_counts.index,
                        color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red', 'Critical': 'darkred'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Download
            csv = result_df.to_csv(index=False)
            st.download_button("📥 Télécharger les résultats", csv, "predictions.csv", "text/csv")

# Page 3: Analyse des Risques
elif page == "📈 Analyse des Risques":
    st.header("📈 Analyse par Profil de Risque")
    
    profiles = create_risk_profiles()
    
    results = []
    for profile in profiles:
        result = service.predict_single_customer(profile['data'])
        if result.get('success'):
            results.append({
                'Profil': profile['name'],
                'Prédiction': 'Churn' if result['churn_prediction'] == 1 else 'Fidèle',
                'Probabilité': result['churn_probability'],
                'Risque': result['churn_risk_level']
            })
    
    if results:
        df_results = pd.DataFrame(results)
        
        # Graphique
        fig = px.bar(df_results, x='Profil', y='Probabilité', 
                    title="Probabilité de Churn par Profil",
                    color='Risque',
                    color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red', 'Critical': 'darkred'},
                    text_auto='.1%')
        st.plotly_chart(fig, use_container_width=True)
        
        # Tableau
        st.dataframe(df_results, use_container_width=True)

# Page 4: Informations
else:
    st.header("ℹ️ Informations sur les Modèles")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Modèles Chargés")
        info = service.get_model_info()
        for model in info['available_models']:
            st.write(f"- {model}")
        
        st.markdown(f"### Nombre de Features")
        st.metric("Features attendues", info['feature_count'])
    
    with col2:
        st.markdown("### Feature Importance (Top 10)")
        importance_df = service.get_feature_importance(10)
        if not importance_df.empty:
            st.dataframe(importance_df, use_container_width=True)
        else:
            st.write("Non disponible")
    
    st.markdown("### 📊 Performance des Modèles")
    st.image("../figures/ROC_Curve___XGBoost.png", caption="Courbe ROC - XGBoost")
    st.image("../figures/Confusion_Matrix___XGBoost.png", caption="Matrice de Confusion")

st.markdown("---")
st.caption("Churn Prediction Dashboard - Projet ML Retail")