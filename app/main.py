import streamlit as st
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import cloudpickle
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile
import os

# Load preprocessor, model, and label encoder
@st.cache_resource
def load_resources():
    with open("new_preprocessor.pkl", 'rb') as f:
        preprocessor = cloudpickle.load(f)
    model = load_model("checkpoint.h5")
    label_encoder = cloudpickle.load(open("label_encoder.pkl", 'rb'))
    return preprocessor, model, label_encoder

preprocessor, model, label_encoder = load_resources()

# Define relevant landmark extraction
def extract_pose_landmarks(image):
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            pose_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
            return pose_array
        else:
            return None

# UI Layout
st.title("🧘 Yoga Pose Classification App")
st.write("Upload an image of a person performing a yoga pose, and this app will classify it!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    pose_vector = extract_pose_landmarks(image)

    if pose_vector is not None:
        # Format as string and series
        pose_vector_str = np.array2string(pose_vector, separator=' ', suppress_small=True)
        pose_vector_series = pd.Series([pose_vector_str])
        
        # Preprocess and predict
        processed = preprocessor.transform(pose_vector_series)
        prediction = model.predict(processed)
        predicted_class = np.argmax(prediction)
        pose_name = label_encoder.inverse_transform([predicted_class])[0]

        st.success(f"🧘 Predicted Pose: **{pose_name}**")
    else:
        st.warning("No human pose detected in the image. Please try another image.")
