import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import streamlit as st
from PIL import Image, ImageOps
import numpy as np

# Set page title and layout
st.set_page_config(
    page_title="AI Image Recognizer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Function to load model
@st.cache(allow_output_mutation=True)
def load_model():
    model = MobileNetV2(weights="imagenet")
    return model

# Define image size
IMAGE_SIZE = 224

# Function to make predictions
def predict(image, model):
    size = (IMAGE_SIZE, IMAGE_SIZE)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image = np.asarray(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction

# Function for UI layout
def run_app():
    st.sidebar.image("George_Brown_College_logo.svg.png", use_column_width=True)
    st.sidebar.header("About")
    st.sidebar.info(
        "This application is a demonstration of how to use "
        "pre-trained models for image classification tasks using Streamlit and TensorFlow. "
        "It uses the MobileNetV2 model, which is trained on the ImageNet dataset. "
        "The app will predict the class of the uploaded image out of 1000 classes. "
    )
    
    st.sidebar.header("Team Members")
    st.sidebar.text(
        """
        - Pablo Guinea Benito
        - Joy
        - Abdullah
        - Dushyant
        """
    )
    
    model = load_model()

    st.title("AI Image Recognizer")
    st.header("Predict the class of an uploaded image")

    file = st.file_uploader("Please upload an image", type=["jpg", "png", "jpeg"])

    if file is not None:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        predictions = predict(image, model)
        class_name = decode_predictions(predictions, top=1)
        st.success(f"This image is most likely a: {class_name[0][0][1]}")

        # Show balloons after submitting the image
        st.balloons()

if __name__ == "__main__":
    run_app()
