import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# 1. Initialize MediaPipe Hand Landmarker
# ============================================================

MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Error: hand_landmarker.task was not found!")
    print("Place the model file in the same folder as this script.")
    raise SystemExit

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = vision.HandLandmarker.create_from_options(options)


# ============================================================
# 2. Initialize Webcam
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    landmarker.close()
    raise SystemExit

ret, frame = cap.read()

if not ret:
    print("Error: Could not read from webcam.")
    cap.release()
    landmarker.close()
    raise SystemExit

frame = cv2.flip(frame, 1)

height, width, _ = frame.shape

canvas = np.zeros((height, width, 3), dtype=np.uint8)


# ============================================================
# 3. Drawing Settings
# ============================================================

# OpenCV uses BGR color format
RED = (0, 0, 255)
BLUE = (255, 0, 0)

# Default color
current_color = RED

brush_thickness = 8

previous_point = None
stroke_points = []


# ============================================================
# 4. Hand Landmark Connections
# ============================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# ============================================================
# 5. Helper Functions
# ============================================================

def get_landmark_point(landmarks, index, image_width, image_height):
    """Convert normalized coordinates into pixel coordinates."""

    x = int(landmarks[index].x * image_width)
    y = int(landmarks[index].y * image_height)

    x = max(0, min(image_width - 1, x))
    y = max(0, min(image_height - 1, y))

    return x, y


def is_finger_up(landmarks, tip_id, pip_id):
    """Check whether a finger is raised."""

    return landmarks[tip_id].y < landmarks[pip_id].y


def get_fingers_state(landmarks):
    """
    Return:
    [Thumb, Index, Middle, Ring, Pinky]

    1 = raised
    0 = folded
    """

    # Detect thumb using the distance between thumb landmarks.
    # This is a simple approximation.
    thumb_tip = landmarks[4]
    thumb_joint = landmarks[3]
    thumb_base = landmarks[2]

    thumb_up = (
        abs(thumb_tip.x - thumb_base.x) > 0.08
        and abs(thumb_tip.x - thumb_joint.x) > 0.025
    )

    index_up = is_finger_up(landmarks, 8, 6)
    middle_up = is_finger_up(landmarks, 12, 10)
    ring_up = is_finger_up(landmarks, 16, 14)
    pinky_up = is_finger_up(landmarks, 20, 18)

    return [
        int(thumb_up),
        int(index_up),
        int(middle_up),
        int(ring_up),
        int(pinky_up)
    ]


def draw_hand_landmarks(image, landmarks):
    """Draw the hand skeleton and landmark points."""

    points = []

    for landmark in landmarks:
        x = int(landmark.x * image.shape[1])
        y = int(landmark.y * image.shape[0])
        points.append((x, y))

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            image,
            points[start],
            points[end],
            (0, 255, 255),
            2
        )

    for point in points:
        cv2.circle(
            image,
            point,
            4,
            (255, 0, 255),
            -1
        )


def recognize_shape(points):
    """Recognize a basic shape from drawing points."""

    if len(points) < 10:
        return "Not enough points"

    contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return "Unknown"

    approximation = cv2.approxPolyDP(
        contour,
        0.04 * perimeter,
        True
    )

    vertices = len(approximation)

    x, y, w, h = cv2.boundingRect(contour)

    if w == 0 or h == 0:
        return "Unknown"

    aspect_ratio = w / float(h)

    if vertices == 3:
        return "Triangle"

    elif vertices == 4:
        if 0.85 <= aspect_ratio <= 1.15:
            return "Square"
        return "Rectangle"

    else:
        area = cv2.contourArea(contour)
        circularity = 0

        if perimeter > 0:
            circularity = 4 * math.pi * area / (perimeter * perimeter)

        if circularity > 0.65:
            return "Circle"

        return "Unknown"


# ============================================================
# 6. Main Loop
# ============================================================

print("\n====================================")
print("         MAGIC HAND PAINTER")
print("====================================")
print("Index finger only: Draw")
print("Index + Middle fingers: Select RED")
print("Thumbs up: Select BLUE")
print("Open palm: Clear canvas")
print("D: Recognize shape")
print("C: Clear canvas")
print("S: Save drawing")
print("Q: Quit")
print("====================================\n")

start_time = time.time()

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("Could not read webcam frame.")
        continue

    frame = cv2.flip(frame, 1)

    image_height, image_width, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp_ms = int((time.time() - start_time) * 1000)

    results = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    gesture_message = "Show your hand"
    drawing_now = False

    # ========================================================
    # 7. Gesture Recognition
    # ========================================================

    if results.hand_landmarks:

        landmarks = results.hand_landmarks[0]

        draw_hand_landmarks(frame, landmarks)

        fingers = get_fingers_state(landmarks)

        thumb = fingers[0]
        index = fingers[1]
        middle = fingers[2]
        ring = fingers[3]
        pinky = fingers[4]

        index_x, index_y = get_landmark_point(
            landmarks,
            8,
            image_width,
            image_height
        )

        # ----------------------------------------------------
        # Gesture 1: Open palm -> Clear canvas
        # ----------------------------------------------------

        if all(finger == 1 for finger in fingers):

            canvas[:] = 0
            stroke_points.clear()

            previous_point = None

            gesture_message = "Canvas Cleared"

        # ----------------------------------------------------
        # Gesture 2: Index + Middle -> Select RED
        # ----------------------------------------------------

        elif (
            index == 1
            and middle == 1
            and ring == 0
            and pinky == 0
        ):

            current_color = RED
            previous_point = None

            gesture_message = "RED Selected"

        # ----------------------------------------------------
        # Gesture 3: Thumb up -> Select BLUE
        # ----------------------------------------------------

        elif (
            thumb == 1
            and index == 0
            and middle == 0
            and ring == 0
            and pinky == 0
        ):

            current_color = BLUE
            previous_point = None

            gesture_message = "BLUE Selected"

        # ----------------------------------------------------
        # Gesture 4: Index finger only -> Draw
        # ----------------------------------------------------

        elif (
            index == 1
            and middle == 0
            and ring == 0
            and pinky == 0
        ):

            drawing_now = True

            gesture_message = "Drawing"

            current_point = (index_x, index_y)

            # Draw a dot
            cv2.circle(
                canvas,
                current_point,
                brush_thickness // 2,
                current_color,
                -1
            )

            # Connect the current point to the previous point
            if previous_point is not None:

                cv2.line(
                    canvas,
                    previous_point,
                    current_point,
                    current_color,
                    brush_thickness
                )

            # Save points for shape recognition
            if (
                not stroke_points
                or math.dist(stroke_points[-1], current_point) > 2
            ):
                stroke_points.append(current_point)

            previous_point = current_point

        # ----------------------------------------------------
        # Other gestures -> Stop drawing
        # ----------------------------------------------------

        else:

            previous_point = None
            gesture_message = "Gesture Not Recognized"

    else:

        previous_point = None

    # ========================================================
    # 8. Overlay Canvas on Webcam
    # ========================================================

    gray_canvas = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGR2GRAY
    )

    _, mask = cv2.threshold(
        gray_canvas,
        10,
        255,
        cv2.THRESH_BINARY
    )

    mask_inv = cv2.bitwise_not(mask)

    frame_background = cv2.bitwise_and(
        frame,
        frame,
        mask=mask_inv
    )

    canvas_foreground = cv2.bitwise_and(
        canvas,
        canvas,
        mask=mask
    )

    output = cv2.add(
        frame_background,
        canvas_foreground
    )

    # ========================================================
    # 9. Display Current Color and Instructions
    # ========================================================

    if current_color == RED:
        color_name = "RED"
    else:
        color_name = "BLUE"

    cv2.putText(
        output,
        f"Color: {color_name}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        current_color,
        2
    )

    cv2.putText(
        output,
        gesture_message,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        "Index: Draw | Two fingers: Red | Thumb: Blue",
        (20, image_height - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        "D: Shape | C: Clear | S: Save | Q: Quit",
        (20, image_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )

    cv2.imshow("Magic Hand Painter", output)

    # ========================================================
    # 10. Keyboard Controls
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    # D -> Recognize shape
    if key == ord('d') or key == ord('D'):

        shape = recognize_shape(stroke_points)

        print("Recognized Shape:", shape)

        cv2.putText(
            output,
            f"Shape: {shape}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow("Magic Hand Painter", output)
        cv2.waitKey(1000)

    # C -> Clear canvas
    elif key == ord('c') or key == ord('C'):

        canvas[:] = 0
        stroke_points.clear()
        previous_point = None

        print("Canvas cleared.")

    # S -> Save drawing
    elif key == ord('s') or key == ord('S'):

        filename = "magic_hand_painting.png"

        cv2.imwrite(filename, canvas)

        print(f"Drawing saved as: {filename}")

    # Q -> Quit
    elif key == ord('q') or key == ord('Q'):

        print("Exiting Magic Hand Painter...")
        break


# ============================================================
# 11. Release Resources
# ============================================================

cap.release()
cv2.destroyAllWindows()
landmarker.close()

print("Program closed successfully")