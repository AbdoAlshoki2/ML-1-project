# ML-1-project

## Overview

Going back to traditional machine learning projects was far from my mind, but the more you learn, the more you realize how much you don't know.

This project is part of my ITI AI track diploma. I built a simple machine learning model to detect hand gestures in real-time—a basic project you can find anywhere.

However, this time I learned things I had never done before, such as saving preprocessing steps, organizing ML training notebooks, and easily reusing models with their corresponding trained preprocessing.

## Code Structure

```
ML-1-project/
├── data/
│   └── hand_landmarks_data.csv    # Training dataset
├── media/                        # Demo videos and images
├── models/                       # Trained machine learning models (.joblib)
├── processors/                   # Saved preprocessing (LabelEncoder, etc.)
├── config.yaml                   # Project configuration
├── ml_project.ipynb              # Training pipeline and exploration
├── streamlit_demo.py             # Real-time UI detector
├── terminal_demo.py              # CLI version of the detector
├── transformers.py               # Custom data transformation logic
├── requirements.txt              # Project dependencies
└── README.md                     # This file
```

## Demo Video

[![Demo Video](media/demo_static_image.png)](media/demo.mp4)

