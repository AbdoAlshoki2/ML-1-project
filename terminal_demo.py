import mediapipe as mp
import cv2
import numpy as np
import joblib
import yaml
from pathlib import Path

from transformers import HandCentering, HandNormalization


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    
    active_model_name = config['active_model']
    model_config = config['models'][active_model_name]
    
    print(f"Loading model: {model_config['name']}")
    
    feature_processor = joblib.load(config['processors']['feature_processor'])
    model = joblib.load(model_config['path'])
    
    label_encoder = None
    if model_config['use_label_encoder']:
        label_encoder = joblib.load(config['processors']['label_encoder'])
        print("Label encoder loaded")
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    
    display = config['display']
    font = getattr(cv2, display['font'])

    
    cap = cv2.VideoCapture(0)
    
    print("Starting hand detection... Press 'q' to quit")
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        numpy_list = np.zeros((1, 63))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                
                for i, landmark in enumerate(hand_landmarks.landmark):
                    numpy_list[0, i * 3] = landmark.x
                    numpy_list[0, i * 3 + 1] = landmark.y
                    numpy_list[0, i * 3 + 2] = landmark.z
                
                feature_transformed = feature_processor.transform(numpy_list)
                prediction = model.predict(feature_transformed)
                
                if label_encoder is not None:
                    prediction = label_encoder.inverse_transform(prediction)
                
                # Stack labels vertically for multiple hands
                y_offset = display['position'][1] + (hand_idx * 40)
                label_text = f"Hand {hand_idx + 1}: {prediction[0]}"
                
                cv2.putText(
                    frame,
                    label_text,
                    (display['position'][0], y_offset),
                    font,
                    display['font_scale'],
                    tuple(display['color']),
                    display['thickness']
                )
        
        cv2.imshow("Hand Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
