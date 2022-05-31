"""
THIS IS NOT IMPLEMENTED YET - the Iris detection needs to be manually built within
the mediapipe package. TBD
"""

import mediapipe as mp
import cv2

class IrisDetector:
    def __init__(self):
        self.image = None
        self.mpDraw = mp.solutions.drawing_utils
        self.mpDrawStyles = mp.solutions.drawing_styles

        self.iris = mp.solutions.iris.Iris()

    def load_image(self, image):
        self.image = image

    def get_iris_data(self):
        """
        The function takes an image, converts it to RGB, passes it to the iris detector,
        and then draws the iris landmarks.
        """

        # To improve performance, mark the image as not writeable to pass by reference.
        self.image.flags.writeable = False
        imageRGB = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        results = self.iris.process(imageRGB)

        self.image.flags.writeable = True
        
        if results.face_landmarks_with_iris:
            for landmark in results.face_landmarks_with_iris:
                self.mpDraw.draw_landmarks(self.image, landmark.landmark_list, self.mpDrawStyles.FACE_LANDMARKS_EYES_AND_IRIS)