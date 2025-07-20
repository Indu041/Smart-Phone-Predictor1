import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import os

# Sample synthetic dataset
data = {
    'brand': ['Samsung', 'Apple', 'Xiaomi', 'Realme', 'Apple', 'Xiaomi'],
    'ram': [4, 6, 4, 8, 8, 6],
    'rom': [64, 128, 64, 128, 256, 128],
    'camera': [12, 48, 16, 64, 108, 48],
    'battery': [4000, 3500, 4500, 5000, 4500, 4300],
    'price': [15000, 60000, 12000, 18000, 70000, 20000]
}
df = pd.DataFrame(data)

# Encode brand
df['brand'] = df['brand'].astype('category').cat.codes

X = df[['brand', 'ram', 'rom', 'camera', 'battery']]
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
os.makedirs('model', exist_ok=True)
pickle.dump(model, open('model/price_model.pkl', 'wb'))
