"""
Streamlit app for real-time hand detection with dynamic model selection.
"""
import streamlit as st
import cv2
import numpy as np
import joblib
import yaml
import mediapipe as mp
from transformers import HandCentering, HandNormalization


@st.cache_resource
def load_config():
    """Load configuration from YAML file."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_feature_processor(path):
    """Load and cache feature processor."""
    return joblib.load(path)


def load_model_and_encoder(model_config, processors_config):
    """Load model and optionally label encoder."""
    model = joblib.load(model_config['path'])
    label_encoder = None
    if model_config['use_label_encoder']:
        label_encoder = joblib.load(processors_config['label_encoder'])
    return model, label_encoder


def main():
    st.set_page_config(page_title="Hand Detection", layout="wide")
    st.title("🖐️ Real-Time Hand Detection")
    
    # Load configuration
    config = load_config()
    models_config = config['models']
    
    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    model_names = list(models_config.keys())
    model_display_names = [models_config[name]['name'] for name in model_names]
    
    selected_idx = st.sidebar.selectbox(
        "Choose a model:",
        range(len(model_names)),
        format_func=lambda i: model_display_names[i]
    )
    selected_model_key = model_names[selected_idx]
    
    # Display settings
    st.sidebar.header("Display Settings")
    font_scale = st.sidebar.slider("Font Scale", 0.5, 2.0, 1.0, 0.1)
    thickness = st.sidebar.slider("Text Thickness", 1, 5, 2)
    color_option = st.sidebar.selectbox("Text Color", ["Green", "Blue", "Red", "Yellow", "White"])
    
    color_map = {
        "Green": (0, 255, 0),
        "Blue": (255, 0, 0),
        "Red": (0, 0, 255),
        "Yellow": (0, 255, 255),
        "White": (255, 255, 255)
    }
    text_color = color_map[color_option]
    
    if 'current_model' not in st.session_state or st.session_state.current_model != selected_model_key:
        with st.spinner(f"Loading {models_config[selected_model_key]['name']}..."):
            model, label_encoder = load_model_and_encoder(
                models_config[selected_model_key],
                config['processors']
            )
            st.session_state.model = model
            st.session_state.label_encoder = label_encoder
            st.session_state.current_model = selected_model_key
            st.success(f"✅ Loaded {models_config[selected_model_key]['name']}")
    
    feature_processor = load_feature_processor(config['processors']['feature_processor'])
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Feed")
        frame_placeholder = st.empty()
    
    with col2:
        st.subheader("Controls")
        run = st.checkbox("Start Detection", value=True)
        st.info(f"**Current Model:** {models_config[selected_model_key]['name']}")
        
        if models_config[selected_model_key]['use_label_encoder']:
            st.info("🏷️ Using label encoder")
        
        prediction_display = st.empty()
    
    cap = cv2.VideoCapture(0)
    
    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam")
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            predictions = []
            
            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    
                    numpy_list = np.zeros((1, 63))
                    for i, landmark in enumerate(hand_landmarks.landmark):
                        numpy_list[0, i * 3] = landmark.x
                        numpy_list[0, i * 3 + 1] = landmark.y
                        numpy_list[0, i * 3 + 2] = landmark.z
                    
                    feature_transformed = feature_processor.transform(numpy_list)
                    prediction = st.session_state.model.predict(feature_transformed)
                    
                    if st.session_state.label_encoder is not None:
                        prediction = st.session_state.label_encoder.inverse_transform(prediction)
                    
                    prediction_text = str(prediction[0])
                    predictions.append(prediction_text)
                    
                    y_offset = 50 + (hand_idx * 40)
                    label_text = f"Hand {hand_idx + 1}: {prediction_text}"
                    
                    cv2.putText(
                        frame,
                        label_text,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        text_color,
                        thickness
                    )
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            if predictions:
                prediction_display.success(f"**Detected:** {', '.join(predictions)}")
            else:
                prediction_display.info("No hands detected")
    
    cap.release()


if __name__ == "__main__":
    main()
