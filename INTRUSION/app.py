import streamlit as st
import pickle
import numpy as np

# Function to load the saved model from disk
def load_model(path=r'C:\Users\IICET 22\Desktop\abhishek\INTRUSION\intrusion_detection_model.pkl'):
    with open(r"C:\Users\IICET 22\Desktop\abhishek\INTRUSION\intrusion_detection_model.pkl", 'rb') as f:
        model = pickle.load(f)  # Load the model using pickle
    return model

# Function to predict intrusion
def predict_intrusion(model, input_data):
    if isinstance(input_data, list):
        input_data = np.array(input_data)

    if input_data.ndim == 1:
        input_data = input_data.reshape(1, -1)

    predictions = model.predict(input_data)

    if isinstance(predictions[0], str):
        return predictions.tolist()

    labels = ['normal', 'anomaly']
    return [labels[p] for p in predictions]

# ---------------- Streamlit UI ----------------
st.title("🚨 Intrusion Detection System (IDS)")

# Load model
model = load_model()

st.write("Enter feature values to check whether the connection is **Normal** or **Anomaly**")

# Number of features (replace 41 if your model trained with different size)
num_features = st.number_input("Number of Features (must match training)", min_value=1, max_value=100, value=10)

# User input fields
input_data = []
for i in range(num_features):
    val = st.number_input(f"Feature {i+1}", value=0.0, format="%.4f")
    input_data.append(val)

# Prediction button
if st.button("Predict"):
    result = predict_intrusion(model, input_data)
    if result[0] == "normal":
        st.success(f"✅ Prediction: {result[0].upper()}")
    else:
        st.error(f"⚠️ Prediction: {result[0].upper()}")
