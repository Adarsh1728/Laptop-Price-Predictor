import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Load model and DataFrame
st.set_page_config(page_title="Laptop Price Predictor", layout="centered")

@st.cache_resource
def load_model():
    try:
        pipe = pickle.load(open("pipe.pkl", "rb"))
        df = pickle.load(open("df.pkl", "rb"))
        return pipe, df
    except Exception as e:
        st.error(f"❌ Error loading model or data: {e}")
        return None, None

Pipe, df = load_model()

st.title("💻 Laptop Price Predictor")

if Pipe is not None and df is not None:
    # User Inputs
    company = st.selectbox('Brand', df['Company'].unique())
    type_ = st.selectbox('Type', df['TypeName'].unique())
    ram = st.selectbox('RAM (in GB)', sorted(df['Ram'].unique()))
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
    os_ = st.selectbox('Operating System', df['Os'].unique())

    if st.button('💰 Predict Price'):
        try:
            # Resolution and PPI
            X_res, Y_res = map(int, resolution.split('x'))
            ppi = ((X_res**2 + Y_res**2) ** 0.5) / screen_size

            # Boolean encoding
            touchscreen_val = 1 if touchscreen == 'Yes' else 0
            ips_val = 1 if ips == 'Yes' else 0

            # Create input DataFrame
            query = pd.DataFrame([[
                company, type_, ram, memory, weight,
                touchscreen_val, ips_val, ppi, cpu, gpu, os_
            ]], columns=df.columns)

            # Show user inputs
            st.subheader("🔍 Input Configuration")
            st.dataframe(query)

            # Predict
            prediction = Pipe.predict(query)[0]
            price = np.exp(prediction)  # Reverse log

            st.success(f"💸 Predicted Price: ₹{price:,.0f}")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
