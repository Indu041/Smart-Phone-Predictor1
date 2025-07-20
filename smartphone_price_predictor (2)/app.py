import streamlit as st
import pickle
import numpy as np

# Load model
model, le = pickle.load(open('model/price_model.pkl', 'rb'))

st.title("📱 Smartphone Price Predictor")

brand = st.selectbox("Brand", ["Samsung", "Xiaomi", "Apple", "Realme", "OnePlus"])
ram = st.slider("RAM (GB)", 2, 16, step=1)
storage = st.slider("Storage (GB)", 32, 512, step=32)
camera = st.slider("Camera (MP)", 8, 108, step=4)
battery = st.slider("Battery (mAh)", 2000, 6000, step=100)

# Encode brand
brand_encoded = le.transform([brand])[0]

# Predict
if st.button("Predict Price"):
    features = np.array([[brand_encoded, ram, storage, camera, battery]])
    prediction = model.predict(features)[0]
    st.success(f"📱 Estimated Price: ₹{int(prediction)}")
