import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Your class_labels and other code...
st.write("Current Working Directory:", os.getcwd())

# CIFAR-10 class labels
class_labels = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# Use Streamlit's cache_resource decorator to load the model once.
@st.cache_resource
def load_the_model():
    try:
        # Correctly load the Keras model using TensorFlow's built-in function
        model = tf.keras.models.load_model(r"C:\Users\IICET 22\Desktop\abhishek\project store\deep learning 2\cifar10_model.h5")
        return model
    except FileNotFoundError:
        st.error("Error: The file 'cifar10_model.h5' was not found. Please make sure the path is correct.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.stop()

# Call the function to load the model and assign it to a variable
model = load_the_model()

# Streamlit app title
st.title("CIFAR-10 Image Classifier")
st.write("Upload an image, and the model will predict its class.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for the model
    # The model was trained on 32x32 pixel images
    image_resized = image.resize((32, 32))
    # Convert image to numpy array and normalize pixel values to [0, 1]
    image_array = np.array(image_resized) / 255.0
    # Reshape the array to match the model's expected input shape (batch_size, height, width, channels)
    image_array = image_array.reshape(1, 32, 32, 3)

    # Make a prediction using the loaded model
    prediction = model.predict(image_array)
    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(prediction)
    # Get the human-readable class label
    predicted_class = class_labels[predicted_class_index]

    # Display the final prediction to the user
    st.subheader(f"Predicted Class: {predicted_class}")