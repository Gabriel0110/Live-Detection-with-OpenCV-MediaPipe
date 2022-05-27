import cv2
import mediapipe as mp
import numpy as np
import pybboxes as pbx
import time
from google.protobuf.json_format import MessageToDict

from hands_detector import HandsDetector

if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    hands = None

    # Choose if you want to detect hands or face
    DETECT_HANDS = True
    DETECT_FACE = True

    if DETECT_HANDS:
        hands = HandsDetector()
    
    cv2_img_coords = []
    face_bounding_box_coords = {}

    while True:
        success, image = cam.read()

        if DETECT_HANDS:
            hands.load_image(image)
            hands_data, hands_type = hands.get_hand_data()

            for hand_data, hand_type in zip(hands_data, hands_type):
                if hands.PINCH_COMMAND_ON:
                    hands.pinch_check()
                if hands.BOUNDING_BOX_ON:
                    hands.bounding_box(hand_type)
                hands.update_landmark_positions(hand_data, hand_type)

        if DETECT_FACE:
            with mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5) as face_detection:

                # To improve performance, mark the image as not writeable to pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_detection.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.detections:
                    for detection in results.detections:
                        face_bounding_box_coords = MessageToDict(detection)['locationData']['relativeBoundingBox']

                        H, W, _ = image.shape
                        ymin = face_bounding_box_coords['ymin']
                        xmin = face_bounding_box_coords['xmin']
                        width = face_bounding_box_coords['width']
                        height = face_bounding_box_coords['height']

                        # Covert the YOLO bounding box coordinates to the original image dimensions
                        converted_coords = pbx.convert_bbox((xmin, ymin, width, height), from_type="yolo", to_type="voc", image_width=W, image_height=H)
                        x = converted_coords[0]+42
                        y = converted_coords[1]+42
                        w = converted_coords[2]+42
                        h = converted_coords[3]+42
                        cv2_img_coords = [x, y, w, h]

                        # Create a bounding box around the face
                        cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)                    

        cv2.imshow("Output", image)

        if cv2.waitKey(1) & 0xFF == ord('p'):
            new_img = image[cv2_img_coords[1]:cv2_img_coords[3], cv2_img_coords[0]:cv2_img_coords[2]]
            cv2.imwrite(f'face_detection{str(time.time()).split(".")[0]}.jpg', new_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()