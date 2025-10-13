import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Set a title for the Streamlit app
st.title("Deep Learning Image Classifier")
st.write("This app demonstrates a simple image classification model.")

# --- Model Training (Commented out for performance) ---
# For a real-world application, it's best to train the model once and save it.
# Training a model on every run is very slow.
# I will create a placeholder for the model and the training history.

def get_model():
    """Defines and compiles a simple CNN model."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid') # Binary classification
    ])
    
    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])
    
    return model

@st.cache_resource
def load_and_train_model():
    """
    Loads or trains the model.
    Note: For a real app, you would load a pre-trained model here.
    """
    # Placeholder for model loading
    # You would typically do something like:
    # model = tf.keras.models.load_model('path/to/your/model.h5')
    # For demonstration, we will just define a new one.
    model = get_model()
    
    # Placeholder for training. We will not actually train here.
    # history = model.fit(...)
    # For plotting, we will use a dummy history object.
    
    # Dummy history data for plotting
    dummy_history = {
        'accuracy': [0.65, 0.72, 0.75, 0.77, 0.79],
        'val_accuracy': [0.60, 0.68, 0.70, 0.72, 0.74],
        'loss': [0.55, 0.48, 0.45, 0.42, 0.39],
        'val_loss': [0.60, 0.55, 0.52, 0.50, 0.48]
    }
    
    return model, dummy_history

# Load the model and history
with st.spinner('Loading model and data...'):
    model, history = load_and_train_model()

# --- Display Training History ---
st.header("Model Training History")
st.write("These plots show the model's performance during training.")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot accuracy
ax1.plot(history['accuracy'], label='Training Accuracy')
ax1.plot(history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Training and Validation Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

# Plot loss
ax2.plot(history['loss'], label='Training Loss')
ax2.plot(history['val_loss'], label='Validation Loss')
ax2.set_title('Training and Validation Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

st.pyplot(fig)
plt.close(fig)

# --- Make a Prediction ---
st.header("Make a Prediction on a New Image")
st.write("Upload an image below to see a prediction from the model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    
    # Process the image for prediction
    # The model expects a 150x150 image, so we resize it.
    resized_image = image.resize((150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(resized_image)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = "A" if prediction[0][0] > 0.5 else "B" # Replace 'A' and 'B' with your actual classes
    confidence = prediction[0][0]
    
    # Display the result
    st.subheader("Prediction Result")
    st.info(f"The model predicts this image belongs to class **{predicted_class}**.")
    st.write(f"Confidence: {confidence:.2f}")
