from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from numpy import ndarray
import globals

emotion_model = load_model(r'Emotion_model.h5')

# Function to retrieve Emotion from Frame


def get_emotions_from_roi():
    try:
        roi_emotion = cv2.resize(globals.roi, (48, 48))  # Resizing
        image_pixels = image.img_to_array(roi_emotion)
        image_pixels = np.expand_dims(image_pixels, axis=0)
        image_pixels /= 255  # Scaling
    except:
        print('Unable to perform ROI')

    try:
        emotional_prediction = emotion_model.predict(
            image_pixels)  # Model prediction
        # Taking maximum value out of emotions
        max_index: ndarray[int] = np.argmax(emotional_prediction[0])
    except:
        print('Unable to perform emotion prediction')

    try:
        emotion = ('Angry', 'Disgusted', 'Fear', 'Happy',
                   'Neutral', 'Sad', 'Surprised')
        predicted_emotion = emotion[max_index]
        globals.emotions.append(predicted_emotion)
    except:
        print('Emotional malfunction')

    return globals.emotions
