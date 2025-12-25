import cv2
import mediapipe as mp
import pyautogui
# import dotenv  # not used in this snippet

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR (OpenCV) -> RGB (MediaPipe)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(image_rgb)

    # default gesture
    hand_gesture = 'other'

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # draw landmarks on the original frame (BGR)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # get normalized y positions (0..1). Smaller y => higher on the image.
            index_finger_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            thumb_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y

            # add a small threshold to avoid jitter
            thresh = 0.02
            if index_finger_y < thumb_y - thresh:
                hand_gesture = 'pointing up'
            elif index_finger_y > thumb_y + thresh:
                hand_gesture = 'pointing down'
            else:
                hand_gesture = 'other'

            # react to gesture
            if hand_gesture == 'pointing up':
                # Try 'volumeup' - might be OS dependent
                pyautogui.press('volumeup')
            elif hand_gesture == 'pointing down':
                pyautogui.press('volumedown')

    frame = cv2.resize(frame, (1280, 1000))   # width, height
    # show the frame regardless of whether a hand was detected
    cv2.imshow('Hand Gesture', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
