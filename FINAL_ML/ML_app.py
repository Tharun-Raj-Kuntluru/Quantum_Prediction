import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Quantum Task Time Predictor",
    page_icon="⚛️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Caching Functions for Performance ---
# @st.cache_data is used for caching data loading/processing.
@st.cache_data
def generate_training_data(num_samples=5000, filename="quantum_task_data_large.csv"):
    """
    Generates and saves a synthetic dataset.
    This function will only run once and the result will be cached.
    """
    if os.path.exists(filename):
        # If file exists, load it directly to save time
        data = pd.read_csv(filename)
        return data
        
    # If file doesn't exist, create it
    qubits = np.random.randint(10, 100, size=num_samples)
    depth = np.random.randint(20, 200, size=num_samples)
    shots = np.random.randint(1000, 10000, size=num_samples)
    cnot_gates = np.random.randint(5, 50, size=num_samples)
    
    base_time = 15
    time_val = (
        base_time + (qubits ** 1.5 * 0.02) + (depth ** 1.2 * 0.05) +
        (shots * 0.001) + (cnot_gates * depth * 0.001) +
        np.random.normal(0, 5, size=num_samples)
    )
    time_val = np.maximum(time_val, 1.0)
    
    data = pd.DataFrame({
        'num_qubits': qubits, 'circuit_depth': depth,
        'num_shots': shots, 'num_cnot_gates': cnot_gates,
        'completion_time': time_val
    })
    data.to_csv(filename, index=False)
    return data

# @st.cache_resource is used for caching non-serializable objects like models.
@st.cache_resource
def train_model(data):
    """
    Trains an XGBoost Regressor model.
    This function will only run once and the trained model will be cached.
    """
    df = data
    features = ['num_qubits', 'circuit_depth', 'num_shots', 'num_cnot_gates']
    target = 'completion_time'
    
    X = df[features]
    y = df[target]
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=1000, learning_rate=0.05,
        max_depth=5, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1
    )
    model.fit(X, y)
    return model

# --- Load Data and Train Model ---
# This part of the script runs once, and the results are cached.
training_data = generate_training_data()
model = train_model(training_data)

# --- Streamlit App Interface ---
st.title("⚛️ Quantum Task Completion Time Predictor")
st.markdown("Use the sliders in the sidebar to input your quantum task parameters. The model will predict the execution time.")

# --- Sidebar for User Input ---
st.sidebar.header("Task Parameters")

qubits_input = st.sidebar.slider(
    'Number of Qubits', 
    min_value=10, max_value=100, value=50, step=1
)
depth_input = st.sidebar.slider(
    'Circuit Depth', 
    min_value=20, max_value=200, value=100, step=5
)
shots_input = st.sidebar.slider(
    'Number of Shots', 
    min_value=1000, max_value=10000, value=4096, step=128
)
cnot_input = st.sidebar.slider(
    'Number of CNOT Gates', 
    min_value=5, max_value=50, value=25, step=1
)

# --- Prediction and Output ---
if st.sidebar.button('**Predict Completion Time**', type="primary"):
    # Create a DataFrame from the user's input
    user_data = pd.DataFrame({
        'num_qubits': [qubits_input],
        'circuit_depth': [depth_input],
        'num_shots': [shots_input],
        'num_cnot_gates': [cnot_input]
    })
    
    # Make a prediction
    with st.spinner('Running prediction...'):
        time.sleep(1) # Simulate a short delay
        predicted_time = model.predict(user_data)[0]
    
    st.subheader("Prediction Result")
    # Use st.metric for a nice, clear display
    st.metric(
        label="Predicted Completion Time", 
        value=f"{predicted_time:.2f} seconds"
    )
    
    st.success("Prediction complete!")

else:
    st.info("Adjust the parameters in the sidebar and click the predict button.")