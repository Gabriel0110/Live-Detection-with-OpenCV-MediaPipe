import mediapipe as mp
import cv2
import pybboxes as pbx
from google.protobuf.json_format import MessageToDict

class FaceDetector:
    def __init__(self):
        self.image = None
        self.mpDraw = mp.solutions.drawing_utils
        self.mpDrawStyles = mp.solutions.drawing_styles

        self.face = mp.solutions.face_detection.FaceDetection()

        self.cv2_img_coords = []
        self.face_bounding_box_coords = {}

    def load_image(self, image):
        self.image = image

    def get_face_data(self):
        """
        The function takes an image, converts it to RGB, passes it to the face detector,
        and then draws a bounding box around the face.
        """

        # To improve performance, mark the image as not writeable to pass by reference.
        self.image.flags.writeable = False
        imageRGB = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        results = self.face.process(imageRGB)

        self.image.flags.writeable = True
        
        if results.detections:
            for detection in results.detections:
                self.face_bounding_box_coords = MessageToDict(detection)['locationData']['relativeBoundingBox']

                H, W, _ = self.image.shape
                ymin = self.face_bounding_box_coords['ymin']
                xmin = self.face_bounding_box_coords['xmin']
                width = self.face_bounding_box_coords['width']
                height = self.face_bounding_box_coords['height']

                # Covert the YOLO bounding box coordinates to the original image dimensions
                converted_coords = pbx.convert_bbox((xmin, ymin, width, height), from_type="yolo", to_type="voc", image_width=W, image_height=H, strict=False)
                x = converted_coords[0]+42
                y = converted_coords[1]+42
                w = converted_coords[2]+42
                h = converted_coords[3]+42
                self.cv2_img_coords = [x, y, w, h]

                # Create a bounding box around the face
                cv2.rectangle(self.image, (x, y), (w, h), (0, 255, 0), 2)

    def set_model_confidence(self, value):
        self.face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=value)