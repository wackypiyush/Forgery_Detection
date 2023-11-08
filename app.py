import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import gdown
import os

# Function to download the models from Google Drive
def download_models():
    forgery_model_url = 'https://drive.google.com/uc?id=1TqE1fc7wLud8-Iga2Ydkz7yfctfo2t52'
    overwriting_model_url = 'https://drive.google.com/uc?id=1XFRY1TXual_CAahUFtQW333XyzRhU9D1'

    gdown.download(forgery_model_url, 'digital_forgery_detection_model3.h5', quiet=False)
    gdown.download(overwriting_model_url, 'overwriting_detection_model2.h5', quiet=False)

# Load the pre-trained models
download_models()
forgery_model = load_model('digital_forgery_detection_model3.h5')
overwriting_model = load_model('overwriting_detection_model2.h5')

# Function to preprocess the uploaded image
def preprocess_image(file_path, target_size=(80, 240)):
    try:
        img = load_img(file_path, target_size=target_size)
        img_array = img_to_array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        st.error(f"Error processing image: {file_path}, {e}")
        return None

# Function to make predictions using the forgery detection model
def detect_digital_forgery(image):
    # Your forgery detection logic goes here
    # Replace this line with your actual model prediction
    prediction = forgery_model.predict(image)
    return prediction

# Function to make predictions using the overwriting detection model
def detect_overwriting(image):
    # Your overwriting detection logic goes here
    # Replace this line with your actual model prediction
    prediction = overwriting_model.predict(image)
    return prediction

# Main Streamlit app
def main():
    st.title("Document Manipulation Detection App")

    # File uploader for images
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image with its real size
        image = preprocess_image(uploaded_file)
        pil_image = Image.open(uploaded_file)
        st.image(pil_image, caption="Uploaded Image", use_column_width=True)

        # Make predictions when the button is clicked
        if st.button("Detect Forgery"):
            forgery_result = detect_digital_forgery(image)
            overwriting_result = detect_overwriting(image)

            # Display the results in a table
            st.table([
                {"": "Accuracy of model on train sets", "Overwriting": "93%", "Digital Forgery": "68%"},
                {"": "Accuracy of model on validation/test sets", "Overwriting": "90%", "Digital Forgery": "64%"},
                {"": "Results/Probabilities", "Overwriting": overwriting_result[0, 0], "Digital Forgery": forgery_result[0, 0]},
                {"": "Presence", "Overwriting": "Overwriting is present" if overwriting_result[0, 0] > 0.5 else "No Overwriting",
                 "Digital Forgery": "Digital Forgery is present" if forgery_result[0, 0] > 0.5 else "No Digital Forgery"}
            ])

# Run the app
if __name__ == "__main__":
    main()
