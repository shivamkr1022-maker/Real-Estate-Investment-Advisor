import sys
import os

# Add the project root directory and src to Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from preprocessing import load_and_clean_data

# Page Configuration
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide"
)

@st.cache_resource
def get_trained_models():
    # Load dataset & train models on the fly for the dashboard
    df = load_and_clean_data("india_housing_prices.csv")
    
    features = [
        'BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt', 
        'Year_Built', 'Floor_No', 'Total_Floors', 'Age_of_Property',
        'Nearby_Schools', 'Nearby_Hospitals', 'Property_Type', 'City',
        'Furnished_Status', 'Parking_Space'
    ]
    features = [f for f in features if f in df.columns]
    X = pd.get_dummies(df[features], drop_first=True)
    
    # Train Regression
    y_reg = df['Future_Price_5Y']
    X_train_r, _, y_train_r, _ = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    reg_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    reg_model.fit(X_train_r, y_train_r)
    
    # Train Classification
    y_clf = df['Good_Investment']
    X_train_c, _, y_train_c, _ = train_test_split(X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
    clf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf_model.fit(X_train_c, y_train_c)
    
    return reg_model, clf_model, X.columns, df

reg_model, clf_model, model_columns, df = get_trained_models()

# App Title & Header
st.title("🏠 Real Estate Investment Advisor")
st.markdown("### Predicting Property Profitability & Future Value using Machine Learning")
st.markdown("---")

# Sidebar User Inputs
st.sidebar.header("📝 Enter Property Details")

city = st.sidebar.selectbox("City", df['City'].unique())
property_type = st.sidebar.selectbox("Property Type", df['Property_Type'].unique())
bhk = st.sidebar.slider("BHK (Bedrooms)", 1, 5, 2)
size_sqft = st.sidebar.number_input("Property Size (SqFt)", min_value=300, max_value=10000, value=1200)
price_lakhs = st.sidebar.number_input("Current Price (in Lakhs)", min_value=10.0, max_value=1000.0, value=75.0)
year_built = st.sidebar.number_input("Year Built", min_value=1980, max_value=2026, value=2015)
furnished = st.sidebar.selectbox("Furnished Status", df['Furnished_Status'].unique() if 'Furnished_Status' in df.columns else ['Unfurnished'])
parking = st.sidebar.selectbox("Parking Space", df['Parking_Space'].unique() if 'Parking_Space' in df.columns else ['Yes'])

# Compute derived features
current_year = 2026
age_of_property = current_year - year_built
price_per_sqft = price_lakhs / size_sqft if size_sqft > 0 else 0

# Main Panel layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Property Summary")
    st.info(f"**Location:** {city}")
    st.info(f"**Type:** {property_type} ({bhk} BHK)")
    info_df = pd.DataFrame({
        "Metric": ["Size (SqFt)", "Current Price (Lakhs)", "Price per SqFt", "Property Age"],
        "Value": [f"{size_sqft} SqFt", f"₹ {price_lakhs} Lakhs", f"₹ {price_per_sqft:.2f}", f"{age_of_property} Years"]
    })
    st.table(info_df)

with col2:
    st.subheader("🔮 Investment & Price Forecast")
    
    if st.button("Evaluate Property Investment", type="primary"):
        # Construct input dataframe
        input_data = pd.DataFrame({
            'BHK': [bhk],
            'Size_in_SqFt': [size_sqft],
            'Price_in_Lakhs': [price_lakhs],
            'Price_per_SqFt': [price_per_sqft],
            'Year_Built': [year_built],
            'Floor_No': [2],
            'Total_Floors': [5],
            'Age_of_Property': [age_of_property],
            'Nearby_Schools': [5],
            'Nearby_Hospitals': [3],
            'Property_Type': [property_type],
            'City': [city],
            'Furnished_Status': [furnished],
            'Parking_Space': [parking]
        })
        
        # Align with model training columns (One-Hot Encoding)
        input_encoded = pd.get_dummies(input_data)
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
        
        # Predictions
        future_price_pred = reg_model.predict(input_encoded)[0]
        investment_pred = clf_model.predict(input_encoded)[0]
        
        # Display Results
        if investment_pred == 1:
            st.success("✅ **Classification: GOOD INVESTMENT!** This property shows strong potential for high returns.")
        else:
            st.warning("⚠️ **Classification: MODERATE / RISKY INVESTMENT.** Please review pricing and locality trends before buying.")
            
        st.metric(label="📈 Estimated Price After 5 Years", value=f"₹ {future_price_pred:.2f} Lakhs")