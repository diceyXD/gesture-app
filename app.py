import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model("gesture_model.h5")

# 🔴 REPLACE with your printed class_names from Colab
labels = [
    "palm","l","fist","fist_moved","thumb",
    "index","ok","palm_moved","c","down"
]

st.title("🖐️ Hand Gesture Recognition")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    img = np.array(image)
    img = cv2.resize(img, (64,64)) / 255.0
    img = np.reshape(img, (1,64,64,3))

    pred = model.predict(img)
    class_id = np.argmax(pred)
    confidence = np.max(pred)

    st.success(f"Prediction: {labels[class_id]}")
    st.info(f"Confidence: {confidence:.2f}")