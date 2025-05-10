# Depression Detection from Text, Image & Speech using Deep Learning Algorithm
## Introduction

This project aims to develop a multimodal system that detects signs of depression using text, image, and speech inputs by leveraging deep learning techniques. The system integrates Natural Language Processing (NLP), Convolutional Neural Networks (CNN), and Recurrent Neural Networks (RNN) to analyze user data and predict depression levels.


## Features

- **Text Analysis**: Uses NLP and sentiment analysis to evaluate user-written text.
- **Image Analysis**: Employs CNNs to analyze facial expressions.
- **Speech Analysis**: Extracts voice tone, pitch, and pauses to determine depressive speech patterns.
- **Fusion Layer**: Combines predictions from all modalities for final classification.

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- Librosa
- Scikit-learn
- Natural Language Toolkit (NLTK)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/depression-detection.git
   cd depression-detection
2.**Create and activate a virtual environment**:
  python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3.**Install dependencies**:
pip install -r requirements.txt
## Dataset Sources:
Text Dataset (Suicide Watch – Reddit comments labeled for depression):

🔗 https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch?select=Suicide_Detection.csv

Image Dataset (Facial Emotion Recognition):

🔗 https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer

Speech Dataset (RAVDESS Emotional Speech Audio):

🔗 https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
## description:
Text: User transcripts or responses.

Image: Facial expression images collected during interviews.

Speech: Audio recordings with depressive and non-depressive speech.

Datasets should be placed in the data/ directory structured by modality.
## How to Run
1. **Train individual model**:
   python src/text_model.py
python src/image_model.py
python src/speech_model.py
2.**Run the fusion model**:
   python src/fusion_model.py
3.**Predict using new data**:
   Provide input in all three formats (text, image, speech), and the system will output a depression classification.
  ## Evaluation Metrics
Accuracy
Precision, Recall
F1 Score
## Final Result:
| **Modality** | **Algorithm**                      | **Accuracy** | **Precision** | **Recall** | **F1 Score** |
| ------------ | ---------------------------------- | ------------ | ------------- | ---------- | ------------ |
| **Image**    | CNN (Convolutional Neural Network) | 97.21%       | High          | High       | High         |
| **Speech**   | CNN                                | 97.56%       | High          | High       | High         |
| **Text**     | Random Forest                      | 95.00%       | High          | High       | High         |

## Future Enhancements
Integration with real-time chatbots.
Deployable mobile/web app.
Larger multimodal datasets.

