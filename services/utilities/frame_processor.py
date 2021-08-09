import time
import cv2
from services.utilities.bpm_processor import bpm_async_calculation
import globals

def process_frame() -> object:
    frame_number = 0
    frame_number += 1

    # try:
    globals.faces_detected = globals.face_haar_cascade_alt.detectMultiScale(globals.stored_frame, minNeighbors=4, scaleFactor=1.2, minSize=(100, 100))
    # Draw the rectangle around each face
    if globals.draw_debug:
        for (x, y, w, h) in globals.faces_detected:  # checking for multiple faces
            # CV2 face_haar_cascade_alt uses BGR -Blue color rectangle
            cv2.rectangle(globals.stored_frame, (x, y),
                          (x + w, y + h), (255, 0, 0), 1)

    # if haar_cascade_alt can't find faces try haar_cascade_default
    if len(globals.faces_detected) <= 0:
        globals.faces_detected = globals.face_haar_cascade_default.detectMultiScale(
            globals.stored_frame, 1.2, 5)

        # Draw the rectangle around each face
        if globals.draw_debug:
            for (x, y, w, h) in globals.faces_detected:  # checking for multiple faces
                # CV2 face_haar_cascade_default uses BGR -Green color rectangle
                cv2.rectangle(globals.stored_frame, (x, y),
                              (x + w, y + h), (0, 255, 0), 1)

    # process every second frame only, 8FPS etc
    if frame_number % 2:
        # move time adding step here should solve the bug
        globals.t1 = time.time() - globals.t0
        globals.times.append(globals.t1)
        globals.executor.submit(bpm_async_calculation())
