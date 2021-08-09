# Original solution that boots a flask driven (local) only
# video streaming solution.

# Orignal solution - flask api allows applications to stream large data efficiently
# partitioned into smaller chunks, over potentially
# long period of time. Ideal for video processing
# https://blog.miguelgrinberg.com/post/video-streaming-with-flask

# Multi-part processing is the key, allowing a chunk in the
# video stream be an image, allowing an prior ML
# processing before actor is presented results

# # References :https://www.youtube.com/watch?v=mzX5oqd3pKA&t=524s

# # Generate frames for live video feed
# def generate_frames():
#     frame_number = 0
#     # Times is used as an array of time keys
# global times
# global data_size
# global forehead
# global forehead_data
# global mean_values
# global frequency
# global time_points
# global bpm_count
# global bpms
# global bpm_history
# global bpm
# global stored_frame
# global faces_detected

# while True:
#     frame_number += 1
#     # read the camera frame
#     success, frame = webCam.read()
#     stored_frame = frame
#     # time appended to times array for data association in loop
#     # clear faces array
#     faces_detected = []
#     # try haar_cascade_alt
#     try:
#         faces_detected = face_haar_cascade_alt.detectMultiScale(frame, minNeighbors=4, scaleFactor=1.2,
#                                                                minSize=(100, 100))
#         # Draw the rectangle around each face
#         if draw_debug:
#             for (x, y, w, h) in faces_detected:  # checking for multiple faces
#                 # CV2 face_haar_cascade_alt uses BGR -Blue color rectangle
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
#     except:
#         print('face_haar_cascade_alt unable to detect face.')
#
#     # if haar_cascade_alt can't find faces try haar_cascade_default
#     if len(faces_detected) <= 0:
#         try:
#             faces_detected = face_haar_cascade_default.detectMultiScale(frame, 1.2, 5)
#             # Draw the rectangle around each face
#             if draw_debug:
#                 for (x, y, w, h) in faces_detected:  # checking for multiple faces
#                     # CV2 face_haar_cascade_default uses BGR -Green color rectangle
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
#         except:
#             print('face_haar_cascade_default unable to detect face.')
#
#     # process every second frame only, 8FPS etc
#     if frame_number % 2:
#         # move time adding step here should solve the bug
#         t1 = time.time() - t0
#         times.append(t1)
#         executor.submit(bpm_async_calculation(frame, faces_detected))
#
#     ret, buffer = cv2.imencode('.jpg', frame)
#     frame = buffer.tobytes()
#
#     # Return the frames to video feed function
#     yield (b'--frame\r\n'
#            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
