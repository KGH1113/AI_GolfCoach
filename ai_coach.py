import cv2
import mediapipe as mp

import socket
hostname = socket.gethostname()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2)

is_first = True # 어드레스 시 첫 프레임을 받아오기 위한 플래그
# 어드레스 시 첫 프레임의 좌표를 저장할 변수
first_center_x, first_center_y, first_radius = None, None, None

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img_h, img_w, _ = img.shape

    img_result = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img)

    mp_drawing.draw_landmarks(
        img_result,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    if results.pose_landmarks:
        # https://google.github.io/mediapipe/solutions/pose.html#pose-landmark-model-blazepose-ghum-3d
        landmark = results.pose_landmarks.landmark

        left_ear_x = landmark[mp_pose.PoseLandmark.LEFT_EAR].x * img_w
        left_ear_y = landmark[mp_pose.PoseLandmark.LEFT_EAR].y * img_h

        right_ear_x = landmark[mp_pose.PoseLandmark.RIGHT_EAR].x * img_w
        right_ear_y = landmark[mp_pose.PoseLandmark.RIGHT_EAR].y * img_h

        center_x = int((left_ear_x + right_ear_x) / 2)
        center_y = int((left_ear_y + right_ear_y) / 2)

        radius = int((left_ear_x - right_ear_x) / 2)
        radius = max(radius, 20)

        if is_first: # 어드레스 시 첫 프레임의 머리 좌표 저장
            first_center_x = center_x
            first_center_y = center_y
            first_radius = int(radius * 2)

            is_first = False
        else:
            cv2.circle(img_result, center=(first_center_x, first_center_y),
                radius=first_radius, color=(0, 255, 255), thickness=2)

            color = (0, 255, 0) # 초록색

            # 머리가 원래 위치보다 많이 벗어난 경우
            if center_x - radius < first_center_x - first_radius \
                or center_x + radius > first_center_x + first_radius:
                color = (0, 0, 255) # 빨간색

            cv2.circle(img_result, center=(center_x, center_y),
                radius=radius, color=color, thickness=2)

    cv2.imshow('AI Golf Coach', img_result)

    if cv2.waitKey(1) == ord('q'):
        break

pose.close()
cap.release()



# import cv2
# import mediapipe as mp
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_pose = mp.solutions.pose

# # For static images:
# IMAGE_FILES = []
# BG_COLOR = (192, 192, 192) # gray
# with mp_pose.Pose(
#     static_image_mode=True,
#     model_complexity=2,
#     enable_segmentation=True,
#     min_detection_confidence=0.5) as pose:
#   for idx, file in enumerate(IMAGE_FILES):
#     image = cv2.imread(file)
#     image_height, image_width, _ = image.shape
#     # Convert the BGR image to RGB before processing.
#     results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     if not results.pose_landmarks:
#       continue
#     print(
#         f'Nose coordinates: ('
#         f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
#         f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
#     )

#     annotated_image = image.copy()
#     # Draw segmentation on the image.
#     # To improve segmentation around boundaries, consider applying a joint
#     # bilateral filter to "results.segmentation_mask" with "image".
#     condition = mp.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
#     bg_image = mp.zeros(image.shape, dtype=mp.uint8)
#     bg_image[:] = BG_COLOR
#     annotated_image = mp.where(condition, annotated_image, bg_image)
#     # Draw pose landmarks on the image.
#     mp_drawing.draw_landmarks(
#         annotated_image,
#         results.pose_landmarks,
#         mp_pose.POSE_CONNECTIONS,
#         landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
#     cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
#     # Plot pose world landmarks.
#     mp_drawing.plot_landmarks(
#         results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

# # For webcam input:
# cap = cv2.VideoCapture(0)
# with mp_pose.Pose(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as pose:
#   while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#       print("Ignoring empty camera frame.")
#       # If loading a video, use 'break' instead of 'continue'.
#       continue

#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = pose.process(image)

#     # Draw the pose annotation on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     mp_drawing.draw_landmarks(
#         image,
#         results.pose_landmarks,
#         mp_pose.POSE_CONNECTIONS,
#         landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
#     # Flip the image horizontally for a selfie-view display.
#     cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
#     if cv2.waitKey(5) & 0xFF == 27:
#       break
# cap.release()