import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Load model and data
Pipe = None
if os.path.exists("pipe.pkl"):
    try:
        with open("pipe.pkl", "rb") as f:
            Pipe = pickle.load(f)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print("❌ pipe.pkl not found.")

# Load the dataframe
try:
    df = pickle.load(open("df.pkl", "rb"))
except Exception as e:
    st.error(f"❌ Error loading df.pkl: {e}")
    st.stop()

st.title("💻 Laptop Price Predictor")

# User inputs
company = st.selectbox('Brand', df['Company'].unique())
type_ = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32])
memory = st.selectbox('Storage Type', df['Memory'].unique())
weight = st.number_input("Weight (in kg)", min_value=0.5, max_value=5.0, step=0.1)
touchscreen = st.selectbox('Touchscreen', ['Yes', 'No'])
ips = st.selectbox('IPS Display', ['Yes', 'No'])
screen_size = st.slider('Screen Size (inches)', 10.0, 18.0, 13.3)
resolution = st.selectbox('Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160',
    '3200x1800', '2880x1800', '2560x1600',
    '2560x1440', '2304x1440'])
cpu = st.selectbox('CPU Brand', df['cpu_brand'].unique())
gpu = st.selectbox('GPU Brand', df['Gpu_brand'].unique())
os = st.selectbox('Operating System', df['Os'].unique())

if st.button('💰 Predict Price'):
    if Pipe is None:
        st.error("❌ Model not loaded. Cannot make prediction.")
    else:
        try:
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
                                          'Touchscreen', 'Ips', 'Ppi', 'cpu_brand', 'Gpu_brand', 'Os'])

            # Prediction
            pred = Pipe.predict(query)[0]
            predicted_price = np.exp(pred)
            st.success(f"💸 The predicted price of this configuration is ₹{predicted_price:,.0f}")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
