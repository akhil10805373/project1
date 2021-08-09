from router import gender_controller, bpm_controller, emotion_controller, age_controller, page_controller
from services.utilities.frame_processor import process_frame
import globals

from flask import Flask
from flask_socketio import SocketIO, emit
from services.models.user import UserData
from threading import Lock
import cv2
import numpy as np

# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.
async_mode = None

app: Flask = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, logger=True, engineio_logger=True, debug=True)
thread = None
thread_lock = Lock()
user = UserData

# ROUTER IMPORTS
# from router import gender_controller
app.add_url_rule('/', view_func=page_controller.index)
app.add_url_rule('/get-age', view_func=age_controller.return_age_data)
app.add_url_rule('/get-emotion', view_func=emotion_controller.return_emotion_data)
app.add_url_rule('/get-gender', view_func=gender_controller.return_gender_data)
app.add_url_rule('/heartbeat', view_func=bpm_controller.heartbeat)
app.add_url_rule('/privacy', view_func=page_controller.privacy)

globals.init()

# Initialise application data processing
@socketio.event
def handle_frame(image):
    nparr = np.fromstring(image, np.uint8)
    globals.stored_frame = cv2.imdecode(nparr, cv2.COLOR_HLS2RGB)
    process_frame()

if __name__ == '__main__':
    socketio.run(app, debug=True, host="0.0.0.0")
