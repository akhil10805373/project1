# Function to retrieve Age from Frame
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
import globals

def getAgesFromRoi():
    try:
        ages = []
        # Resize to 200*200 as model is trained on this size
        globals.roi = cv2.resize(globals.roi, (200, 200))
        image_pixels = image.img_to_array(globals.roi)
        image_pixels = np.expand_dims(image_pixels, axis=0)
        image_pixels /= 255  # Scaling

        img = image_pixels.reshape(-1, 200, 200, 3)

        age = globals.age_model.predict(img)  # Predicting age
        # print(int(age))
        age = int(age)
        ages.append(age)  # Appending age
    except:
        print('Age extraction issue')
    return ages
