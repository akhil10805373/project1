from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import globals

gender_model = load_model(r'gender_model.h5')


# Function to get Gender from frame
def get_genders_from_roi():
    genders = []
    globals.roi = cv2.resize(globals.roi, (200, 200))  # Resizing image
    
    image_pixels = image.img_to_array(globals.roi)
    image_pixels = np.expand_dims(image_pixels, axis=0)
    image_pixels /= 255  # Scaling the images

    predictions = gender_model.predict(image_pixels)  # Predicting gender
    # If the predictions close to 0 then Male else Female
    if predictions < 0.3:
        genders.append('Male')
    else:
        genders.append('Female')
    return genders
