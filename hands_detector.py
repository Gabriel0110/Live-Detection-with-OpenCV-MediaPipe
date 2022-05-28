import mediapipe as mp
import cv2
import numpy as np

class HandsDetector:
    def __init__(self):
        self.image = None
        self.mpDraw = mp.solutions.drawing_utils
        self.mpDrawStyles = mp.solutions.drawing_styles

        self.hands = mp.solutions.hands.Hands()
        self.left_hand_landmarks = {
            "thumb_tip_position": None,
            "index_tip_position": None,
            "middle_tip_position": None,
            "ring_tip_position": None,
            "pinky_tip_position": None,
            "wrist_position": None,
            }
        self.right_hand_landmarks = {
            "thumb_tip_position": None,
            "index_tip_position": None,
            "middle_tip_position": None,
            "ring_tip_position": None,
            "pinky_tip_position": None,
            "wrist_position": None,
            }

        # Settings to turn on/off commands
        self.INDEX_TOUCH_COMMAND_ON = True
        self.PINCH_COMMAND_ON = False
        self.BOUNDING_BOX_ON = True

    def load_image(self, image):
        self.image = image

    def reset_landmarks(self):
        """
        This function resets the landmarks of the left and right hand to None
        """

        self.left_hand_landmarks = {
            "thumb_tip_position": None,
            "index_tip_position": None,
            "middle_tip_position": None,
            "ring_tip_position": None,
            "pinky_tip_position": None,
            "wrist_position": None,
            }
        self.right_hand_landmarks = {
            "thumb_tip_position": None,
            "index_tip_position": None,
            "middle_tip_position": None,
            "ring_tip_position": None,
            "pinky_tip_position": None,
            "wrist_position": None,
            }

    def bounding_box(self, hand_type):
        """
        Draw a bounding box around the hand(s)
        
        :param hand_type: The type of hand you want to draw the bounding box around
        """

        # Draw a rectangle with cv2 around the hand if the positions are not None
        # The text is opposite of what mediapipe says because it's inverted
        if hand_type == "Left":
            left_thumb = self.left_hand_landmarks["thumb_tip_position"]
            left_wrist = self.left_hand_landmarks["wrist_position"]
            left_pinky = self.left_hand_landmarks["pinky_tip_position"]
            left_middle = self.left_hand_landmarks["middle_tip_position"]
            if all(value != None for value in self.left_hand_landmarks.values()):
                cv2.rectangle(self.image, (left_thumb[0] - 10, left_wrist[1] + 10), (left_pinky[0] + 10, left_middle[1] - 10), (0, 255, 0), 2)
                cv2.putText(self.image, "Right Hand", (left_middle[0] - 80, left_middle[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if hand_type == "Right":
            right_thumb = self.right_hand_landmarks["thumb_tip_position"]
            right_wrist = self.right_hand_landmarks["wrist_position"]
            right_pinky = self.right_hand_landmarks["pinky_tip_position"]
            right_middle = self.right_hand_landmarks["middle_tip_position"]
            if all(value != None for value in self.right_hand_landmarks.values()):
                cv2.rectangle(self.image, (right_thumb[0] - 10, right_wrist[1] + 10), (right_pinky[0] + 10, right_middle[1] - 10), (0, 255, 0), 2)
                cv2.putText(self.image, "Left Hand", (right_middle[0] - 80, right_middle[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def pinch_check(self):
        """
        If the distance between the thumb and index finger tips is less than 10 pixels, exit the program
        """

        left_thumb = self.left_hand_landmarks["thumb_tip_position"]
        right_thumb = self.right_hand_landmarks["thumb_tip_position"]
        left_index = self.left_hand_landmarks["index_tip_position"]
        right_index = self.right_hand_landmarks["index_tip_position"]

        # Calculate the distance between the thumb and index finger tips and draw a line between them
        if left_thumb and left_index:
            cv2.line(self.image, left_thumb, left_index, (0, 0, 255), 2)

            # Exit if euclidean distance between the thumb and index finger tips is less than 10 pixels
            dist = np.linalg.norm(np.array(left_thumb) - np.array(left_index))
            if dist < 10:
                print("PINCH DETECTED - exiting...")
                exit()

        if right_thumb and right_index:
            cv2.line(self.image, right_thumb, right_index, (0, 0, 255), 2)

            # Exit if euclidean distance between the thumb and index finger tips is less than 10 pixels
            dist = np.linalg.norm(np.array(right_thumb) - np.array(right_index))
            if dist < 10:
                print("PINCH DETECTED - exiting...")
                exit()

    def index_tip_touch_check(self):
        """
        If the distance between both index fingers is less than 10 pixels, exit the program
        """
        left_index = self.left_hand_landmarks["index_tip_position"]
        right_index = self.right_hand_landmarks["index_tip_position"]

        # Calculate the distance between both index finger tips and draw a line between them
        if left_index and right_index:
            cv2.line(self.image, left_index, right_index, (0, 0, 255), 2)

            # Exit if euclidean distance between both index finger tips is less than 10 pixels
            dist = np.linalg.norm(np.array(left_index) - np.array(right_index))
            if dist < 10:
                print("INDEX TIP TOUCH DETECTED - exiting...")
                exit()
    
    def update_landmark_positions(self, hand_data, hand_type):
        """
        This function takes in the hand data and hand type and updates the hand landmarks dictionary
        with the new positions of the hand landmarks.
        
        :param hand_data: a list of tuples containing the landmark index, x, and y coordinates
        :param hand_type: "Left" or "Right"
        """

        for (idx, cx, cy) in hand_data:
            if idx == mp.solutions.hands.HandLandmark.THUMB_TIP:
                if hand_type == "Left":
                    self.left_hand_landmarks["thumb_tip_position"] = (cx, cy)
                elif hand_type == "Right":
                    self.right_hand_landmarks["thumb_tip_position"] = (cx, cy)

            if idx == mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP:
                if hand_type == "Left":
                    self.left_hand_landmarks["index_tip_position"] = (cx, cy)
                elif hand_type == "Right":
                    self.right_hand_landmarks["index_tip_position"] = (cx, cy)

            if idx == mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP:
                if hand_type == "Left":
                    self.left_hand_landmarks["middle_tip_position"] = (cx, cy)
                elif hand_type == "Right":
                    self.right_hand_landmarks["middle_tip_position"] = (cx, cy)

            if idx == mp.solutions.hands.HandLandmark.RING_FINGER_TIP:
                if hand_type == "Left":
                    self.left_hand_landmarks["ring_tip_position"] = (cx, cy)
                elif hand_type == "Right":
                    self.right_hand_landmarks["ring_tip_position"] = (cx, cy)

            if idx == mp.solutions.hands.HandLandmark.PINKY_TIP:
                if hand_type == "Left":
                    self.left_hand_landmarks["pinky_tip_position"] = (cx, cy)
                elif hand_type == "Right":
                    self.right_hand_landmarks["pinky_tip_position"] = (cx, cy)

            if idx == mp.solutions.hands.HandLandmark.WRIST:
                if hand_type == "Left":
                    self.left_hand_landmarks["wrist_position"] = (cx, cy)
                elif hand_type == "Right":
                    self.right_hand_landmarks["wrist_position"] = (cx, cy)


    def get_hand_data(self):
        """
        It takes an image as input, and returns a list of hand landmarks and a list of hand types (right
        or left)
        
        :param image: The image to be processed
        :return: the hand data and the hand type.
        """

        hands_data = []
        hands_type = []

        # To improve performance, mark the image as not writeable to pass by reference.
        self.image.flags.writeable = False
        imageRGB = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imageRGB)

        # Draw the hand annotations on the image.
        self.image.flags.writeable = True

        # Check if any hands are detected
        if results.multi_hand_landmarks:
            # Get hand landmarks and hand type (right or left)
            for hand in results.multi_handedness:
                handType = hand.classification[0].label
                hands_type.append(handType)
            for hand_landmark in results.multi_hand_landmarks:
                hand_data = []
                for idx, landmark in enumerate(hand_landmark.landmark):
                    image_height, image_width, _ = self.image.shape
                    cx, cy = int(landmark.x * image_width), int(landmark.y * image_height)
                    hand_data.append((idx, cx, cy))
                hands_data.append(hand_data)
                self.mpDraw.draw_landmarks(
                    self.image,
                    hand_landmark,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    self.mpDrawStyles.get_default_hand_landmarks_style(),
                    self.mpDrawStyles.get_default_hand_connections_style())
        else:
            hands_data = []
            hands_type = []
            self.reset_landmarks()

        return hands_data, hands_type