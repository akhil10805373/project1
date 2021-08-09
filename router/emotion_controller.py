from flask import jsonify

from services.utilities import face_detection
from services.utilities.emotion_detection import get_emotions_from_roi
import globals


def return_emotion_data():
    globals.roi = face_detection.region_of_interest_crop()

    if globals.roi is None:
        return jsonify(Emotion=[])

    return jsonify(Emotion=get_emotions_from_roi())
