import cv2
import mediapipe as mp
import math

# Initialize mediapipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)  # Open webcam

#to calculate the angle between three points
def calculate_angle(a, b, c):
    angle = math.degrees(
        math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    )
    if angle < 0:
        angle += 360
    return angle

def count_fingers(hand_landmarks):
    count = 0

    # Thumb check (using angle between 2, 3, 4)
    thumb_angle = calculate_angle(hand_landmarks.landmark[2], hand_landmarks.landmark[3], hand_landmarks.landmark[4])
    if thumb_angle > 180:
        count += 1

    
    if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:              # Index check
        count += 1
    if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y:            # Middle check
        count += 1
    if hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y:            # Ring check
        count += 1
    if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y:            # Pinky check
        count += 1

    return count

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break

    frame = cv2.flip(frame, 1)     # frame fliping horizontally for mirror effect

    # Convert the image to RGB (mediapipe uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            num_fingers = count_fingers(landmarks)
            
            cv2.putText(frame, f'Fingers: {num_fingers}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Hand Tracking - Finger Count', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()