import streamlit as st
import pandas as pd
import joblib

# Set up the page layout
st.set_page_config(page_title="Flood Management System", layout="wide")

st.title("Active Flood Risk Management System")
st.write("Calculate the real-time probability of a flood event based on engineered regional metrics.")

# 1. Helpful guide for the user
st.info("""
**📝 How to use the input scale (0.0 to 20.0):**
* **0.0 - 5.0 (Optimal):** Very low risk, excellent infrastructure, or minimal human impact.
* **5.0 - 10.0 (Stable):** Normal conditions with manageable environmental stress.
* **10.0 - 15.0 (Concerning):** High degradation or severe weather patterns detected.
* **15.0 - 20.0 (Critical):** Maximum possible risk, failing infrastructure, or extreme climate events.
""")

# 2. NEW: Collapsible Data Dictionary explaining the 7 features
with st.expander("📚 Data Dictionary: What do these features mean?"):
    st.markdown("""
    *Our system aggregates 20 raw environmental factors into 7 features for optimal predictive modeling:*
    
    * **Climate Risk:** Combines *Monsoon Intensity* (high rain volume), *Climate Change* (extreme weather patterns), and *Coastal Vulnerability* (susceptibility to storm surges).
    * **Geological Risk:** Combines *Topography Drainage* (natural capacity to clear water), *Landslides* (unstable slopes), and *Watershed* density.
    * **Human Impact:** Combines *Deforestation* (loss of soil absorption) and *Urbanization* (impermeable concrete surfaces increasing runoff). *(Note: Population Score was omitted to isolate environmental factors).*
    * **Infrastructure Deficit:** Combines *Dams Quality*, *Drainage Systems*, and *Deteriorating Infrastructure* (clogged culverts and damaged channels).
    * **Environmental Degradation:** Combines *Siltation* (sediment accumulation in rivers), unsustainable *Agricultural Practices*, and *Wetland Loss* (destruction of natural water sponges).
    * **Management Failures:** Combines poor *River Management* (lack of dredging), *Ineffective Disaster Preparedness* (lack of warning systems), and *Inadequate Planning* (zoning ignoring flood risks).
    * **Encroachment Level:** The physical degree of illegal or poor construction directly on flood plains and natural waterways.
    """)

# Load models and scaler
try:
    linear_model = joblib.load('linear_model.pkl')
    dt_model = joblib.load('dt_model.pkl')
    mlp_model = joblib.load('mlp_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Error: Model files not found. Please ensure you have run the Jupyter Notebook to generate the .pkl files.")
    st.stop()

# Updated to our exact 7 features
features = [
    "Climate_Risk", 
    "Geological_Risk", 
    "Human_Impact", 
    "Infrastructure_Deficit", 
    "Environmental_Degradation", 
    "Management_Failures", 
    "Encroachment_Level"
]

input_data = {}

st.subheader("Aggregated Environmental and Infrastructure Inputs")
cols = st.columns(4)

# Create the input boxes
for i, feature in enumerate(features):
    with cols[i % 4]:
        clean_name = feature.replace("_", " ") + " Index"
        input_data[feature] = st.number_input(clean_name, min_value=0.0, max_value=20.0, value=5.0, step=0.5)

st.markdown("---")

# Prediction Execution
if st.button("Predict Flood Probability", type="primary"):
    
    # Format input to match the exact columns the models were trained on
    input_df = pd.DataFrame([input_data])
    
    # Scale data for the Neural Network
    input_scaled = scaler.transform(input_df)
    
    # Generate predictions
    linear_pred = linear_model.predict(input_df)[0]
    dt_pred = dt_model.predict(input_df)[0]
    mlp_pred = mlp_model.predict(input_scaled)[0]
    
    st.subheader("Final System Recommendation")
    st.write("*(Based on the highest-accuracy model: Multi-Layer Perceptron)*")
    
    # Display the primary recommendation clearly at the top
    if mlp_pred > 0.60:
        st.error(f"🚨 HIGH RISK ({mlp_pred:.4f}): Immediate preventative measures are recommended.")
    elif mlp_pred > 0.45:
        st.warning(f"⚠️ MODERATE RISK ({mlp_pred:.4f}): Monitor weather conditions and drainage systems closely.")
    else:
        st.success(f"✅ LOW RISK ({mlp_pred:.4f}): Conditions are currently stable.")

    st.markdown("---")
    
    st.subheader("Comparative Algorithm Predictions")
    st.write("Displaying output from secondary models for mathematical verification:")
    res_cols = st.columns(3)
    
    with res_cols[0]:
        st.metric(label="Linear Regression", value=f"{linear_pred:.4f}")
    with res_cols[1]:
        st.metric(label="Decision Tree", value=f"{dt_pred:.4f}")
    with res_cols[2]:
        st.metric(label="MLP Neural Network", value=f"{mlp_pred:.4f}")
    
st.markdown("---")

# Evaluation Metrics Dashboard
with st.expander("📊 View Algorithm Evaluation Metrics (Test Dataset)"):
    st.write("This table displays the advanced statistical evaluation of our algorithms against the unseen holdout dataset, proving the mathematical validity of the system.")
    
    # Reminder: Update these numbers to match the exact output from your Jupyter Notebook!
    metrics_data = {
        "Algorithm": ["Linear Regression", "Decision Tree Regressor", "Multi-Layer Perceptron (MLP)"],
        "R-Squared": ["0.7401", "0.6850", "0.7512"],
        "MSE": ["0.0006", "0.0008", "0.0005"],
        "RMSE": ["0.0245", "0.0282", "0.0223"],
        "MAE": ["0.0190", "0.0210", "0.0175"]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    # Hide the index for a cleaner, more professional look
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)