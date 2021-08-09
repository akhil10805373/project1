from flask import jsonify

from services.utilities import face_detection
import globals
from services.utilities.gender_detection import get_genders_from_roi

def return_gender_data():
    globals.roi = face_detection.region_of_interest_crop()

    if globals.roi is None:
        return jsonify(Gender=[])

    return jsonify(Gender=get_genders_from_roi())