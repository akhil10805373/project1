from flask import jsonify

from services.utilities import face_detection
from services.utilities.age_processor import getAgesFromRoi
import globals


def return_age_data():
    try:
        # TODO: Area of breakdown!
        globals.roi = face_detection.region_of_interest_crop()
    except:
        globals.roi = None
        print('Unable to detect face.')

    if globals.roi is None:
        return jsonify(Age=[])

    return jsonify(Age=getAgesFromRoi())
