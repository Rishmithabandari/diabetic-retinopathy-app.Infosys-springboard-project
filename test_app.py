import streamlit as st
import joblib
from PIL import Image
from utils.image_processing import preprocess_image

# Load the trained model
model = joblib.load('models/model.pkl')

st.title("Mediscan - Retinal Image Analysis")

uploaded_file = st.file_uploader("Upload a retinal image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze"):
        with open("temp_image.png", "wb") as f:
            f.write(uploaded_file.getbuffer())

        processed_image = preprocess_image("temp_image.png")

        if processed_image.shape[0] == model.n_features_in_:
            prediction = model.predict([processed_image])
            st.success(f"Predicted Condition: **{prediction[0]}**")
        else:
            st.error("⚠️ Image size mismatch. Please upload a 224x224 image.")
