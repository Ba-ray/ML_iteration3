import cv2 as cv
import mediapipe as mp
import copy
import itertools
import csv
import time
from keypoint_classifier import KeyPointClassifier
import pyautogui
import threading

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # not taking z coordinates

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def logging_csv(number, mode, landmark_list):

    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'keypoint_training.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])

 

d = {0:"up",1:"right",2:"left",3:"down"}


def perform_action_with_delay(value, delay):
    def perform_action():
        # Perform action after delay
        time.sleep(delay)
        if value == 0:
            pyautogui.press('w')
            print("Performed 'w' action")
        elif value == 1:
            pyautogui.press('d')
            print("Performed 'd' action")
        elif value == 2:
            pyautogui.press('a')
            print("Performed 'a' action")
        elif value == 3:
            pyautogui.press('s')
            print("Performed 's' action")

    # Start a new thread for performing the action
    threading.Thread(target=perform_action).start()

#
# Camera preparation ###############################################################
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

# Model load #############################################################
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode='store_true',
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

keypoint_classifier = KeyPointClassifier()

mode =0
number = -1
while True:
    # Process Keys
    key = cv.waitKey(10)
    if key == 27:  # ESC
        break

    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
        mode = 1
    
    if key == 110:  # n
        mode = 0

    # Camera capture
    ret, image = cap.read()
    if not ret:
        break
    image = cv.flip(image, 1)  # Mirror display
    debug_image = copy.deepcopy(image)

    # Detection implementation
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            
            # Landmark calculation
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)

            # Conversion to relative coordinates / normalized coordinates
            pre_processed_landmark_list = pre_process_landmark(landmark_list)

            # print(pre_processed_landmark_list)
            # Write to the dataset file
            logging_csv(number,mode,pre_processed_landmark_list)

            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

            perform_action_with_delay(hand_sign_id, 0.05)

    cv.imshow('Hand Gesture Recognition', debug_image)

cap.release()
cv.destroyAllWindows()