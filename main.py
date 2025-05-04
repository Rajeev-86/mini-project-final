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
import sys

# To import
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(BASE_DIR)

MODEL_PATH = os.path.join(BASE_DIR, 'model.h5')
PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'preprocessor.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder.pkl')
# Load preprocessor, model, and label encoder
@st.cache_resource

def preprocess_first(X: pd.Series) -> np.ndarray:
    return np.vstack(X.apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))).astype(np.float64)

def select_relevant_landmarks(X: np.ndarray, relevant_ids=None) -> np.ndarray:
    if relevant_ids is None:
        relevant_ids = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]

    selected_indices = []
    for idx in relevant_ids:
        selected_indices.extend([idx * 3, idx * 3 + 1, idx * 3 + 2])

    return X[:, selected_indices]

def load_resources():
    with open(PREPROCESSOR_PATH, 'rb') as f:
        preprocessor = cloudpickle.load(f)
        
    model = load_model(MODEL_PATH)
    label_encoder = cloudpickle.load(open(ENCODER_PATH, 'rb'))
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
    st.image(image, caption="Uploaded Image", use_container_width=True)


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
