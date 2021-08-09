import time
from datetime import datetime

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.models import load_model

def init():
    # # List of fields needs to be refactored
    global age_model
    global bpm_count
    global bpm_history
    global bpms
    global current_time
    global data_size
    global draw_debug
    global executor
    global eys
    global face
    global faces
    global face_haar_cascade_alt
    global face_haar_cascade_default
    global faces_detected
    global fft
    global forehead
    global forehead_data
    global frequency
    global landmark_detector
    global last_center
    global latest_bpm
    global mean_values
    global pruned_fft
    global pruned_freqs
    global run_state
    global stored_bpms
    global stored_frame
    global stored_timestamps
    global t0
    global time_points
    global times
    global roi
    global emotions

    # # DOCS https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor
    executor = ThreadPoolExecutor(2)
    age_model = load_model(r'age_model.h5')
    bpm_count = 1
    bpm_history = []
    bpms = []
    current_time = datetime.now().strftime("%H:%M:%S")
    data_size = 256
    draw_debug = True
    eys = np.array([0, 0, 0, 0])
    face = np.array([0, 0, 0, 0])
    faces = []
    face_haar_cascade_alt = cv2.CascadeClassifier("haar_cascade_frontalface_alt.xml")
    face_haar_cascade_default = cv2.CascadeClassifier("haar_cascade_frontalface_default.xml")
    faces_detected = []
    fft = []
    forehead = np.array([0, 0, 0, 0])
    forehead_data = []
    frequency = []
    landmark_detector = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel("lbfmodel.yaml")
    last_center = np.array([0, 0])
    latest_bpm = []
    mean_values = []
    pruned_fft = []
    pruned_freqs = []
    run_state = False
    stored_bpms = []
    stored_frame = None
    stored_timestamps = []
    t0 = time.time()
    time_points = []
    times = []
    emotions = []
