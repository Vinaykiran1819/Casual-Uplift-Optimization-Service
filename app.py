import streamlit as st
import pandas as pd
import numpy as np
from src.causal_uplift_service.pipelines.prediction_pipeline import CustomData, PredictPipeline

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Causal Uplift Service", layout="wide")

st.title("🛍️ Causal Uplift Optimization Service")
st.markdown("""
**Identify the 'Persuadables'.** This tool predicts the **Uplift Score** (Incremental Probability of Purchase) for a specific customer.  
* **High Uplift (> 5%):** Target this user! The email will cause them to buy.  
* **Low/Negative Uplift:** Do not disturb. They are either a "Sure Thing" or a "Lost Cause."
""")

st.sidebar.header("Customer Profile")

# --- INPUT FORM ---
def user_input_features():
    # 1. Behavioral Features
    recency = st.sidebar.slider("Months Since Last Purchase", 1, 12, 5)
    history = st.sidebar.number_input("Total Historical Spend ($)", min_value=0.0, value=200.0)
    
    # 2. Category Usage (Yes/No)
    col1, col2 = st.sidebar.columns(2)
    with col1:
        mens = st.checkbox("Bought Men's Items?", value=True)
        womens = st.checkbox("Bought Women's Items?", value=False)
    with col2:
        newbie = st.checkbox("New Customer?", value=True)
        visit = st.checkbox("Visited Site Recently?", value=False)
    
    # Map Booleans to 1/0
    mens = 1 if mens else 0
    womens = 1 if womens else 0
    newbie = 1 if newbie else 0
    visit = 1 if visit else 0

    # 3. Demographics & Segments
    zip_code = st.sidebar.selectbox("Location (Zip Code Type)", 
                                    ['Urban', 'Suburban', 'Rural'])
    
    channel = st.sidebar.selectbox("Acquisition Channel", 
                                   ['Web', 'Phone', 'Multichannel'])
    
    # History Segment (Must match training data categories exactly)
    history_segment = st.sidebar.selectbox("Spend Category", 
        ['1) $0 - $100', '2) $100 - $200', '3) $200 - $350', 
         '4) $350 - $500', '5) $500 - $750', '6) $750 - $1,000', '7) $1,000 +'])

    # Initialize CustomData Object
    data = CustomData(
        recency=recency,
        history=history,
        mens=mens,
        womens=womens,
        newbie=newbie,
        visit=visit,
        zip_code=zip_code,
        history_segment=history_segment,
        channel=channel
    )
    
    return data

# --- MAIN EXECUTION ---
user_data = user_input_features()

if st.button("Predict Uplift Score"):
    try:
        # 1. Convert Input to DataFrame
        df = user_data.get_data_as_data_frame()
        
        # 2. Show the "Raw" Input for transparency
        st.subheader("Customer Data Input")
        st.dataframe(df)

        # 3. Predict
        pipeline = PredictPipeline()
        uplift_pred = pipeline.predict(df)
        
        # The model returns an array, get the first value
        score = uplift_pred[0] 

        # 4. Display Results
        st.subheader("Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        # Metric Card
        col1.metric("Predicted Uplift", f"{score:.2%}")
        
        # Interpretation Logic
        if score > 0.05:
            col2.success("✅ **Recommendation: SEND EMAIL**")
            col2.write("This user is highly persuadable.")
        elif score > 0:
            col2.warning("⚠️ **Recommendation: MARGINAL**")
            col2.write("Small positive effect. Send only if budget allows.")
        else:
            col2.error("⛔ **Recommendation: DO NOT SEND**")
            col2.write("Email will have no effect or negative effect (Sleeping Dog).")

    except Exception as e:
        st.error(f"An error occurred: {e}")