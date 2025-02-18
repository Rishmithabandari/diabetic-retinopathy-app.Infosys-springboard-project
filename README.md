# MediScan-AI-Powered-Medical-Image-Analysis-for-Disease-Diagnosis_December_2024
Main Branch Checking
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib
import time
import os
import matplotlib.pyplot as plt

# Load Joblib Model
MODEL_PATH = "models/model.pkl"
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found: {MODEL_PATH}")
    st.stop()

model = joblib.load(MODEL_PATH)

# Get model class labels
CLASSES = model.classes_

# Set Expected Image Size (Must match model training input size)
IMAGE_SIZE = int(np.sqrt(model.n_features_in_))  # Auto-detect image size from model

st.markdown("""
    <style>
        /* Change Font for Entire App */
        html, body, [class*="st-"] {
            font-family: 'Poppins', sans-serif;
        }
    </style>
""", unsafe_allow_html=True)


# Custom Styling for UI
st.markdown("""
    <style>
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #1e1e2e !important;
            padding: 10px;
        }
        [data-testid="stSidebarNav"] {
            background-color: #25253a !important;
            color: white !important;
        }
        .sidebar-title {
            font-size: 24px;
            font-weight: bold;
            color: #ffffff;
            text-align: center;
            padding: 10px;
            border-bottom: 2px solid #00c6ff;
        }
        .sidebar-radio label {
            font-size: 18px;
            padding: 10px;
            cursor: pointer;
            color: white !important;
        }
        .sidebar-radio label:hover {
            background-color: #2b2b4f !important;
            border-radius: 10px;
        }

        /* Main Page Styling */
        h1 {
            color: #ffffff !important;
            text-align: center;
        }
        .stButton>button {
            background-color: #008ecc !important;
            color: white !important;
            font-size: 18px !important;
            padding: 10px !important;
            border-radius: 8px !important;
            border: none !important;
        }
        .stButton>button:hover {
            background-color: #008ecc !important;
        }
        .stMarkdown {
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation with Custom Icons
st.sidebar.markdown("<div class='sidebar-title'>🩺 WELCOME TO MEDISCAN</div>", unsafe_allow_html=True)
page = st.sidebar.radio("", ["🏠 Home 🏡", "ℹ️ About 📖", "🔍 Diagnosis 🏥"], key='navigation')

# Home Page
if page == "🏠 Home 🏡":
    st.title("👁️ AI-Based Eye Disease Detection")

    # Load and display the banner image with use_container_width=True
    st.image("IMG1.jpg", use_container_width=True)

    st.markdown("""
        ## 🌟 How it Works:
        - 📸 **Upload an eye image**
        - 🏥 **AI detects Cataracts, Glaucoma, or Diabetic Retinopathy**
        - ⚡ **Instant results with confidence levels**
    """)

    st.button("🚀 Get Started")

# About Page
elif page == "ℹ️ About 📖":
    st.title("📖 About Diabetic Retinopathy & AI Diagnosis")
    st.markdown("""
        ### 🏥 What is Diabetic Retinopathy?
        - Caused by **high blood sugar damaging the retina**.
        - **No early symptoms** – can lead to **blindness**.
        - **Early detection prevents vision loss**.

        ### 🔬 How Our AI Model Works:
        1️⃣ **Upload a retina image**  
        2️⃣ **AI detects disease & severity**  
        3️⃣ **Instant report with confidence levels**
    """)

# Diagnosis Page
elif page == "🔍 Diagnosis 🏥":
    st.title("🔍 AI Diagnosis for Eye Diseases")
    uploaded_file = st.file_uploader("📸 Upload an eye image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="📸 Uploaded Image", use_container_width=True)

        # Convert image to OpenCV format
        image_cv = np.array(image)

        # Convert to grayscale for analysis
        gray_image = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)

        # Apply histogram equalization to enhance contrast
        enhanced_image = cv2.equalizeHist(gray_image)

        # Apply additional processing for severe cases
        blurred_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
        sharp_image = cv2.addWeighted(enhanced_image, 1.5, blurred_image, -0.5, 0)

        # Display the original and enhanced images
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR))
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(sharp_image, cmap="gray")
        axes[1].set_title("Enhanced Image")
        axes[1].axis("off")

        st.pyplot(fig)

        # Process Image for Model Prediction
        image_resized = Image.fromarray(gray_image).resize((IMAGE_SIZE, IMAGE_SIZE))
        image_array = np.array(image_resized) / 255.0
        image_flattened = image_array.flatten().reshape(1, -1)

        if st.button("🔍 Analyze Image"):
            with st.spinner('🔍 AI is analyzing... Please wait!'):
                time.sleep(3)

                # Make Prediction
                probabilities = model.predict_proba(image_flattened)
                prediction_index = np.argmax(probabilities)
                max_confidence = np.max(probabilities) * 100
                predicted_class = CLASSES[prediction_index]

                # Extract Statistical Data for Severity
                mean_intensity = np.mean(sharp_image)
                std_dev_intensity = np.std(sharp_image)

                # Updated Classification with Normal Retina Detection
                if mean_intensity > 150 and std_dev_intensity < 25:
                    severity = "✅ Normal Healthy Retina"
                elif max_confidence < 45 and mean_intensity < 110 and std_dev_intensity > 60:
                    severity = "⚠️ Severe Diabetic Retinopathy"
                elif max_confidence < 60 and mean_intensity < 130 and std_dev_intensity > 50:
                    severity = "🟠 Moderate Diabetic Retinopathy"
                elif max_confidence < 80 and mean_intensity < 150 and std_dev_intensity > 30:
                    severity = "🟡 Mild Diabetic Retinopathy"
                else:
                    severity = "⚠️ No Significant Signs of Diabetic Retinopathy"

              # Display Results
                st.success(f"**Prediction:** {predicted_class}")
                st.info(f"**Severity Level:** {severity}")
              

              # Footer
                st.markdown("""
                  <hr style='border:1px solid gray'>
                  <center>
                  © 2025 Infosys Project by Rishmitha. All rights reserved.
                  </center>
                """, unsafe_allow_html=True)
