import cv2
import time

from hands_detector import HandsDetector
from face_detector import FaceDetector

if __name__ == "__main__":
    cam = cv2.VideoCapture(0)

    # Choose if you want to detect hands or face
    DETECT_HANDS = True
    DETECT_FACE = True

    if DETECT_HANDS:
        hands = HandsDetector()
    if DETECT_FACE:
        face = FaceDetector()

    while True:
        success, image = cam.read()

        if DETECT_HANDS:
            hands.load_image(image)
            hands_data, hands_type = hands.get_hand_data()

            # Do things with the hands (command checking, etc)
            for hand_data, hand_type in zip(hands_data, hands_type):
                if hands.INDEX_TOUCH_COMMAND_ON:
                    hands.index_tip_touch_check()
                if hands.PINCH_COMMAND_ON:
                    hands.pinch_check()
                if hands.BOUNDING_BOX_ON:
                    hands.bounding_box(hand_type)
                hands.update_landmark_positions(hand_data, hand_type)

        if DETECT_FACE:
            face.load_image(image)
            face.get_face_data()

        cv2.imshow("Output", image)

        if cv2.waitKey(1) & 0xFF == ord('p'):
            new_img = image[face.cv2_img_coords[1]:face.cv2_img_coords[3], face.cv2_img_coords[0]:face.cv2_img_coords[2]]
            cv2.imwrite(f'face_detection{str(time.time()).split(".")[0]}.jpg', new_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()