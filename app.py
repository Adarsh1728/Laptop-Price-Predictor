
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Load model and data
Pipe = None  # Initialize as None

if os.path.exists("pipe.pkl"):
    try:
        Pipe = pickle.load(open("pipe.pkl", "rb"))
        print("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
else:
    st.error("❌ pipe.pkl not found.")

df = pickle.load(open("df.pkl", "rb"))

st.title("💻 Laptop Price Predictor")

# Company
company = st.selectbox('Brand', df['Company'].unique())

# Laptop Type
type_ = st.selectbox('Type', df['TypeName'].unique())

# RAM
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32])

# Memory
memory = st.selectbox('Storage Type', df['Memory'].unique())

# Weight
weight = st.number_input("Weight (in kg)", min_value=0.5, max_value=5.0, step=0.1)

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['Yes', 'No'])

# IPS
ips = st.selectbox('IPS Display', ['Yes', 'No'])

# Screen size
screen_size = st.slider('Screen Size (inches)', 10.0, 18.0, 13.3)

# Resolution
resolution = st.selectbox('Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160',
    '3200x1800', '2880x1800', '2560x1600',
    '2560x1440', '2304x1440'])

# CPU
cpu = st.selectbox('CPU Brand', df['cpu_brand'].unique())

# GPU
gpu = st.selectbox('GPU Brand', df['Gpu_brand'].unique())

# Operating System
os = st.selectbox('Operating System', df['Os'].unique())

if st.button('💰 Predict Price'):
    if Pipe is None:
        st.error("🚫 Model not loaded. Cannot make predictions.")
    else:
        # Convert touchscreen and ips to 0/1
        touchscreen_val = 1 if touchscreen == 'Yes' else 0
        ips_val = 1 if ips == 'Yes' else 0

        # Calculate PPI
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

        # Build input DataFrame
        query = pd.DataFrame([[company, type_, ram, memory, weight,
                               touchscreen_val, ips_val, ppi, cpu, gpu, os]],
                             columns=['Company', 'TypeName', 'Ram', 'Memory', 'Weight',
                                      'Touchscreen', 'IPS', 'ppi', 'cpu_brand', 'Gpu_brand', 'Os'])

        try:
            predicted_price = np.exp(Pipe.predict(query)[0])
            st.title("💸 The predicted price of this configuration is ₹{:,.0f}".format(predicted_price))
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
