# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 12:03:37 2025

@author: Freda Erinmwingbovo
"""


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# ------------------------
# App Configuration
# ------------------------
st.set_page_config(
    page_title="Customer Brand Preference Predictor",
    layout="wide",
    page_icon="💻"
)
sns.set_style("whitegrid")

# ------------------------
# Helper Functions
# ------------------------
@st.cache_data
def load_training_data(path="CompleteResponses.csv"):
    return pd.read_csv(path)

def fit_encoders_and_prepare(df_train, education_map, region_map):
    df_train['elevel_label'] = df_train['elevel'].map(education_map)
    df_train['region'] = df_train['zipcode'].map(region_map)

    le_edu = LabelEncoder()
    le_region = LabelEncoder()

    df_train['elevel_encoded'] = le_edu.fit_transform(df_train['elevel_label'])
    df_train['region_encoded'] = le_region.fit_transform(df_train['region'])
    return df_train, le_edu, le_region

def build_or_load_model(X_train, y_train, model_path="RandomForest_BrandPreference.pkl"):
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
    return model

def prepare_uploaded(df_uploaded, le_edu, le_region, education_map, region_map, df_train_prepared):
    # Map education & region
    df_uploaded['elevel_label'] = df_uploaded['elevel'].map(education_map)
    df_uploaded['region'] = df_uploaded['zipcode'].map(region_map)

    # Fill missing mapping with training mode
    edu_mode = df_train_prepared['elevel_label'].mode()[0]
    region_mode = df_train_prepared['region'].mode()[0]
    df_uploaded['elevel_label'] = df_uploaded['elevel_label'].fillna(edu_mode)
    df_uploaded['region'] = df_uploaded['region'].fillna(region_mode)

    # Safe encoding using fitted LabelEncoders
    df_uploaded['elevel_encoded'] = df_uploaded['elevel_label'].apply(
        lambda x: x if x in le_edu.classes_ else edu_mode
    )
    df_uploaded['elevel_encoded'] = le_edu.transform(df_uploaded['elevel_encoded'])

    df_uploaded['region_encoded'] = df_uploaded['region'].apply(
        lambda x: x if x in le_region.classes_ else region_mode
    )
    df_uploaded['region_encoded'] = le_region.transform(df_uploaded['region_encoded'])

    return df_uploaded

# ------------------------
# Fixed mappings
# ------------------------
education_map = {
    0: "Less than High School",
    1: "High School",
    2: "Some College",
    3: "4-Year College",
    4: "Master's/Doctoral/Professional"
}

region_map = {
    0: "New England", 1: "Mid-Atlantic", 2: "East North Central",
    3: "West North Central", 4: "South Atlantic", 5: "East South Central",
    6: "West South Central", 7: "Mountain", 8: "Pacific"
}

# ------------------------
# Sidebar Controls
# ------------------------
st.sidebar.title("💼 Controls")
st.sidebar.markdown("Upload an incomplete survey CSV to predict missing brand preferences.")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
train_button = st.sidebar.button("Train / Refresh Model")

st.sidebar.markdown("---")
st.sidebar.info("""
- CSV columns: salary, age, elevel, car, zipcode, credit  
- Brand prediction: 0 = Acer, 1 = Sony
""")
st.sidebar.markdown("---")

# ------------------------
# Header
# ------------------------
st.title("💻 Customer Brand Preference Predictor")
st.markdown("""
Welcome! Upload your incomplete survey CSV and get predictions for customer computer brand preferences instantly.
""")

# ------------------------
# Load and prepare training data
# ------------------------
try:
    df_train = load_training_data()
    st.info("Training data loaded successfully.")
except Exception as e:
    st.error(f"Could not load CompleteResponses.csv: {e}")
    st.stop()

df_train_prepared, le_edu, le_region = fit_encoders_and_prepare(df_train.copy(), education_map, region_map)
X = df_train_prepared[['salary', 'age', 'elevel_encoded', 'car', 'region_encoded', 'credit']]
y = df_train_prepared['brand']

# ------------------------
# Train or load model
# ------------------------
model_path = "RandomForest_BrandPreference.pkl"
if train_button:
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    joblib.dump(rf, model_path)
    st.success("Model trained and saved.")
else:
    rf = build_or_load_model(X, y, model_path=model_path)

# ------------------------
# Tabs
# ------------------------
tab1, tab2, tab3 = st.tabs(["📊 Predict", "📈 Visualizations", "ℹ️ About / Contact"])

# ------------------------
# Tab 1 - Predict
# ------------------------
with tab1:
    st.header("Predict Brand Preference")
    if uploaded_file:
        df_uploaded = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data Preview")
        st.dataframe(df_uploaded.head())

        df_uploaded = prepare_uploaded(df_uploaded, le_edu, le_region, education_map, region_map, df_train_prepared)
        X_new = df_uploaded[['salary', 'age', 'elevel_encoded', 'car', 'region_encoded', 'credit']]
        preds = rf.predict(X_new)
        df_uploaded['predicted_brand'] = preds

        st.subheader("Predictions (First Rows)")
        st.dataframe(df_uploaded[['salary', 'age', 'elevel_label', 'car', 'region', 'credit', 'predicted_brand']].head())

        csv = df_uploaded.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions CSV", csv, "Predicted_Survey.csv", "text/csv")
    else:
        st.info("Upload a CSV to get predictions.")

# ------------------------
# Tab 2 - Visualizations (Polished & Mobile-Friendly)
# ------------------------
with tab2:
    st.header("Predicted Brand Visualizations")
    
    if uploaded_file:
        st.subheader("Brand Distribution")
        brand_counts = df_uploaded['predicted_brand'].value_counts()
        fig1, ax1 = plt.subplots(figsize=(4,4))  # Mobile-friendly size
        ax1.pie(
            brand_counts,
            labels=['Acer', 'Sony'],
            autopct='%1.1f%%',
            startangle=90,
            colors=['#FF9999','#66B2FF'],
            wedgeprops={'edgecolor':'white'}
        )
        ax1.set_title("Predicted Brand Share")
        st.pyplot(fig1, use_container_width=True)

        st.subheader("Feature Importance")
        if hasattr(rf, "feature_importances_"):
            fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
            fig2, ax2 = plt.subplots(figsize=(4,3))
            fi.plot(kind='barh', ax=ax2, color=['#FFA500' if x==fi.max() else '#87CEEB' for x in fi])
            ax2.set_xlabel("Relative Importance")
            ax2.set_title("Top Features Influencing Prediction")
            st.pyplot(fig2, use_container_width=True)

        st.subheader("Salary vs Age by Predicted Brand")
        fig3, ax3 = plt.subplots(figsize=(5,4))
        sns.boxplot(
            x='predicted_brand',
            y='salary',
            data=df_uploaded,
            palette=['#FF9999','#66B2FF'],
            ax=ax3
        )
        ax3.set_xticklabels(['Acer', 'Sony'])
        ax3.set_title("Salary Distribution by Brand")
        st.pyplot(fig3, use_container_width=True)

        fig4, ax4 = plt.subplots(figsize=(5,4))
        sns.boxplot(
            x='predicted_brand',
            y='age',
            data=df_uploaded,
            palette=['#FF9999','#66B2FF'],
            ax=ax4
        )
        ax4.set_xticklabels(['Acer', 'Sony'])
        ax4.set_title("Age Distribution by Brand")
        st.pyplot(fig4, use_container_width=True)

        st.subheader("Regional Summary Heatmap")
        summary_region = df_uploaded.groupby('region')['predicted_brand'].value_counts().unstack(fill_value=0)
        fig5, ax5 = plt.subplots(figsize=(6,4))
        sns.heatmap(summary_region, annot=True, fmt="d", cmap="YlGnBu", ax=ax5)
        ax5.set_ylabel("Region")
        ax5.set_xlabel("Predicted Brand (0=Acer, 1=Sony)")
        ax5.set_title("Predicted Brand Counts by Region")
        st.pyplot(fig5, use_container_width=True)
        
    else:
        st.info("Upload a CSV file to see interactive visualizations.")


# ------------------------
# Tab 3 - About / Contact
# ------------------------
with tab3:
    st.header("About This App")
    st.markdown("""
    **Customer Brand Preference Predictor** is built with Python, Streamlit, Pandas, and scikit-learn.  
    It predicts a customer's preferred computer brand (Acer or Sony) from survey data.
    """)
    st.subheader("Contact / Portfolio")
    st.markdown("""
- GitHub: [🐙](https://github.com/Freda-Erinmwingbovo)  
- LinkedIn: [🔗](https://www.linkedin.com/in/freda-erinmwingbovo)  
- Portfolio: [🌐](https://yourportfolio.com)  
- Email: engrfreda@gmail.com
""")









