import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('model/price_model.pkl', 'rb'))

st.title("📱 Smartphone Price Predictor")

brand = st.selectbox("Brand", ['Samsung', 'Apple', 'Xiaomi', 'Realme'])
ram = st.slider("RAM (GB)", 2, 16)
rom = st.slider("ROM (GB)", 16, 512)
camera = st.slider("Camera (MP)", 8, 108)
battery = st.slider("Battery (mAh)", 2000, 6000)

brand_map = {'Samsung': 0, 'Apple': 1, 'Xiaomi': 2, 'Realme': 3}
brand_encoded = brand_map[brand]

features = np.array([[brand_encoded, ram, rom, camera, battery]])

if st.button("Predict"):
    price = model.predict(features)[0]
    st.success(f"Estimated Price: ₹{price:,.2f}")
