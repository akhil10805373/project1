import cv2
import numpy as np
from scipy import signal
import globals
from services.utilities import face_detection

def bpm_async_calculation( ):
    # try:
    face_detection.get_primary_face()
    # except:
        # print('BPM:Face Rec Issue')

    # try:
    globals.forehead = face_detection.get_forehead_rect()
    # except:
        # print('Face forehead detection rec issue')

    # Combined forehead area with face cheeks
    # try:
    left_x1, left_x2, right_x1, right_x2, y1, y2 = face_detection.get_face_cheeks()
    x, y, w, h = globals.forehead
    
    forehead_hsv = cv2.cvtColor(globals.stored_frame[y:y + h, x:x + w, :], cv2.COLOR_BGR2HSV)

    HUE = (forehead_hsv[:, :, 0] / 180).reshape(-1, )
    new_HUE = HUE[np.where((HUE > 0) & (HUE < 0.1))]

    left_cheek_hsv = cv2.cvtColor(globals.stored_frame[y1:y2, left_x1:left_x2, :], cv2.COLOR_BGR2HSV)
    right_cheek_hsv = cv2.cvtColor(globals.stored_frame[y1:y2, right_x1:right_x2, :], cv2.COLOR_BGR2HSV)
    
    left_HUE = (left_cheek_hsv[:, :, 0] / 180).reshape(-1, )
    
    right_HUE = (right_cheek_hsv[:, :, 0] / 180).reshape(-1, )
    
    new_left_HUE = left_HUE[np.where((left_HUE > 0) & (left_HUE < 0.1))]
    
    new_right_HUE = right_HUE[np.where((right_HUE > 0) & (right_HUE < 0.1))]
    
    new_cheek_forehead_HUE = np.concatenate((new_left_HUE, new_right_HUE, new_HUE))

    globals.forehead_data.append(new_cheek_forehead_HUE)
    globals.mean_values.append(np.mean(new_cheek_forehead_HUE))
    # except:
    #     globals.times.pop()
    #     print("BPM: can't detect face")

    length = len(globals.forehead_data)

    try:
        if length > globals.data_size:
            # [- var :] acts as a slice shorthand for pointer at X pos
            globals.forehead_data = globals.forehead_data[-globals.data_size:]
            globals.times = globals.times[-globals.data_size:]
            globals.mean_values = globals.mean_values[-globals.data_size:]
            length = globals.data_size
    except:
        print('Failure to obtain: instance one')

    try:
        # now wait until we have 256 frames
        if length >= 256:

            # since we process every two frames now,
            # we can assume that "originally" we have as much as twice the data amount
            # so the length is multiplied by 2, and we can have more time points.
            # After we get more time points, we can use linear interpolation to mock more the sample points then
            new_length = length * 2
            time_gap = globals.times[-1] - globals.times[0]
            fps = new_length / time_gap
            
            globals.frequency = fps / new_length * np.arange(new_length // 2 + 1) * 60.
            globals.time_points = np.linspace(globals.times[0], globals.times[-1], new_length)

            try:
                bpm = use_hsv(length)
            except:
                print('Failed to use HSV')

            if (bpm != 0):
                globals.bpms.append(bpm)

            # narrow the window_range, now returns the last 1 second heartbeat
            window_range = int(fps)

            if len(globals.bpms) > window_range:
                average_bpm = np.mean(globals.bpms)
                globals.bpms = globals.bpms[-window_range:]
                globals.stored_bpms.append(average_bpm)
                globals.latest_bpm.append(average_bpm)
                globals.stored_timestamps.append(globals.time.time())

            if len(globals.stored_bpms) > 300:
                globals.stored_bpms = globals.stored_bpms[-300:]
        
    except:
        print('BPM calc not performed')


# For using the IIR filter, the minimum length of data should be 123, given order of 20 implementation of
# Algorithms for Monitoring Heart Rate and Respiratory Rate From the Video of a User’s Face
# Filtering is done before transforming, however according to the paper, it should be done after transforming.
def use_hsv( length ) -> object:
    if length <= 100:
        # print('lenght')
        # print(length)
        return 0

    # we assume we "originally" have twice the amount than now
    new_length = length * 2
    fs = new_length / (globals.times[-1] - globals.times[0])
    if not np.all(np.diff(globals.times) > 0):
        print('times error')

    # Now we can generate more data points using linear interpolation.
    try:
        interp = np.interp(globals.time_points, globals.times, globals.mean_values)
        balanced_interp = interp - np.mean(np.hamming(new_length) * interp)
    except:
        print('interpolation failed...')

    # change bpm range from 36~198 to 48~198 (0.6*60, 3.3*60)
    sos = signal.iirfilter(N=20, Wn=[0.8, 3.3], fs=fs, output="sos")
    filtered = signal.sosfiltfilt(sos, balanced_interp)
    fft = np.abs(np.fft.rfft(filtered))
    bpm = globals.frequency[np.argmax(fft)]
    return bpm

# # if the person has bangs, we can use cheek area only, to detect whether a person has bangs,
# # we can use azure cognitive api
# def bpm_async_calculation_cheek_only( frame, faces_detected ):
#     global times
#     global data_size
#     global forehead
#     global forehead_data
#     global mean_values
#     global frequency
#     global time_points
#     global bpm_count
#     global bpms
#     global bpm_history
#     global bpm
#     global latest_bpm
#     global stored_bpms
#     try:
#         # face_detection.get_face_rect(frame)
#         face_detection.get_primary_face(faces_detected)
#     except:
#         print('Face Rec Issue')

#     try:
#         forehead = face_detection.get_forehead_rect()
#     except:
#         print('Face forehead detection rec issue')
#     # Combined forehead area with face cheeks
#     try:
#         left_x1, left_x2, right_x1, right_x2, y1, y2 = face_detection.get_face_cheeks(frame)
#         left_cheek_hsv = cv2.cvtColor(frame[y1:y2, left_x1:left_x2, :], cv2.COLOR_BGR2HSV)
#         right_cheek_hsv = cv2.cvtColor(frame[y1:y2, right_x1:right_x2, :], cv2.COLOR_BGR2HSV)
#         left_HUE = (left_cheek_hsv[:, :, 0] / 180).reshape(-1, )
#         right_HUE = (right_cheek_hsv[:, :, 0] / 180).reshape(-1, )
#         new_left_HUE = left_HUE[np.where((left_HUE > 0) & (left_HUE < 0.1))]
#         new_right_HUE = right_HUE[np.where((right_HUE > 0) & (right_HUE < 0.1))]
#         new_cheek_forehead_HUE = np.concatenate((new_left_HUE, new_right_HUE))
#         forehead_data.append(new_cheek_forehead_HUE)
#         mean_values.append(np.mean(new_cheek_forehead_HUE))
#     except:
#         times.pop()
#         print("can't detect face")
#     length = len(forehead_data)
#     try:
#         if length > data_size:
#             # [- var :] acts as a slice shorthand for pointer at X pos
#             forehead_data = forehead_data[-data_size:]
#             times = times[-data_size:]
#             mean_values = mean_values[-data_size:]
#             length = data_size
#     except:
#         print('Failure to obtain: instance one')
#     try:
#         if length >= 256:
#             new_length = length * 2
#             time_gap = times[-1] - times[0]
#             fps = new_length / time_gap
#             frequency = fps / new_length * np.arange(new_length // 2 + 1) * 60.
#             time_points = np.linspace(times[0], times[-1], new_length)
#             try:
#                 bpm = use_hsv(length)
#             except:
#                 print('Failed to use HSV')
#             if (bpm != 0):
#                 bpms.append(bpm)
#             # narrow the window_range, now returns the last 1 second heartbeat
#             window_range = int(fps)
#             if len(bpms) > window_range:
#                 average_bpm = np.mean(bpms)
#                 bpms = bpms[-window_range:]
#                 stored_bpms.append(average_bpm)
#                 latest_bpm.append(average_bpm)
#                 stored_timestamps.append(time.time())
#             if len(stored_bpms) > 300:
#                 stored_bpms = stored_bpms[-300:]
#             global current_time
#         # print("BPM calc performed.")
#     except:
#         print('BPM calc not performed')


# def update_bpm_history( average_bpm, bpm_history ):
#     global current_time
#     # TODO: Extract, explain it's use.
#     try:
#         if (current_time != datetime.now().strftime("%H:%M:%S") and average_bpm > 1):
#             current_time = datetime.now().strftime("%H:%M:%S")
#             newBPM = {'date': datetime.now().isoformat(), 'value': average_bpm}
#             bpm_history.append(newBPM)
#             print(newBPM)
#     except:
#         print("Failure to append BPM to dict")

# # Get the lowest 20% bpm based on the bpms history
# @globals.route('/getLowestRestBpm')
# def get_lowest_rest_bpm():
#     global stored_bpms
#     if len(stored_bpms) < 300:
#         return jsonify("Not enough data for calculating the rest heartrate")
#     return jsonify([np.average(np.sort(stored_bpms)[0:60])])

# # Get the resting bpm based the bpms history
# @globals.route('/getRestBpm')
# def get_rest_bpm():
#     global stored_bpms
#     if len(stored_timestamps) < 1:
#         return "No timestamp data!"
#     if int(stored_timestamps[-1] - stored_timestamps[0]) < 10:
#         return "Not enough data!"
#     heart_ranges = [[46, 50], [51, 55], [56, 60], [61, 65], [66, 70], [71, 75], [76, 80], [81, 85], [86, 90],
#                     [91, 95], [96, 100], [101, 105], [106, 110], [111, 150]]
#     df = pd.DataFrame(stored_bpms)
#     print(df[0].count())
#     rest_bpm_range = None
#     rest_range_count = 0
#     # find the heart range with the most counts
#     for heart_range in heart_ranges:
#         count = df[0].between(heart_range[0], heart_range[1]).sum()
#         if (count > rest_range_count):
#             rest_range_count = count
#             rest_bpm_range = heart_range
#     return jsonify(rest_bpm_range)
