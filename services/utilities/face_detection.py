import numpy as np
import globals


def get_forehead_rect():
    x, y, w, h = globals.face
    newX = int(x + w * 0.5 - (w * 0.4 / 2))
    newY = int(y + h * 0.1 - (h * 0.1 / 2))
    newW = int(0.4 * w)
    newH = int(0.15 * h)
    globals.forehead = np.array([newX, newY, newW, newH])
    return globals.forehead


def get_face_rect():
    faces = list(
        globals.face_haar_cascade_alt.detectMultiScale(globals.stored_frame, minNeighbors=4, scaleFactor=1.2, minSize=(100, 100)))
    if len(faces) > 0:
        faces.sort(key=lambda x: x[-1] * x[-2])
        globals.face = globals.faces[-1]

        shifted = shift(globals.face)
        if shifted > 10:
            globals.face = globals.face

    return globals.face


def get_face_cheeks():
    _, landmarks = globals.landmark_detector.fit(
        globals.stored_frame, np.array([globals.face]))
    landmark = landmarks[0][0]
    left_x1 = int(landmark[4][0])
    left_x2 = int(landmark[20][0])
    y1 = int((landmark[29][1] + landmark[30][1]) / 2)
    y2 = int((landmark[33][1] + landmark[50][1]) / 2)
    right_x1 = int(landmark[23][0])
    right_x2 = int(landmark[12][0])
    return left_x1, left_x2, right_x1, right_x2, y1, y2


'''
fixed serious bug here caused by 'face = faces[-1]'
which means the face will always be the latest face, the method shift() will be be useless, 
and every face rect will differ greatly, resulting in extremely unstable bpms.
this bug is probably caused by merging or extracting the code from the main body.

Also the face rect shown in the page does not equal to the face rect used in the program,
see line 127 & 139 in app.py.
'''


def get_primary_face():
    globals.faces = list(globals.faces_detected)

    if len(globals.faces) > 0:
        globals.faces.sort(key=lambda x: x[-1] * x[-2])

        temp_face = globals.faces[-1]
        shifted = shift(temp_face)

        # print(shifted)
        if shifted > 10:
            globals.face = temp_face

    # print(face)
    return globals.face


def region_of_interest_crop() -> object:
    # For each face detected on frame
    globals.roi = None
    for (x, y, w, h) in globals.faces_detected:
        # cropping region of interest i.e. face area from  image
        globals.roi = globals.stored_frame[y:y + w, x:x + h]
    return globals.roi


def shift(detected):
    x, y, w, h = detected
    center = np.array([x + 0.5 * w, y + 0.5 * h])
    shifted = np.linalg.norm(center - globals.last_center)
    globals.last_center = center
    return shifted
