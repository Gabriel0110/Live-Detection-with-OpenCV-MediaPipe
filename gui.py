from turtle import width
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QPushButton, QCheckBox, QSlider, QGridLayout, QTabWidget
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np

from hands_detector import HandsDetector
from face_detector import FaceDetector


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = False

    def run(self):
        """Runs the thread, grabs frames from camera and emits signal"""
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            result, image = cap.read()
            if result:
                a.process_detections(image)
                self.change_pixmap_signal.emit(image)

        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class App(QWidget):
    def __init__(self):
        super().__init__()

        # Setup everything for detection processing
        self.DETECT_HANDS = True
        self.DETECT_FACE = True

        if self.DETECT_HANDS:
            self.hands = HandsDetector()
        if self.DETECT_FACE:
            self.face = FaceDetector()

        self.setWindowTitle("OpenCV & MediaPipe Detection")
        self.display_width = 640
        self.display_height = 480

        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.display_width, self.display_height)

        # create a text label
        self.textLabel = QLabel('Webcam')

        # create a button to start/stop the webcam
        self.camButton = QPushButton('Start Webcam')
        self.camButton.clicked.connect(self.toggle_webcam)

        # create checkboxes for each detection (hands, face)
        self.faceCheckBox = QCheckBox('Face Detection')
        self.faceCheckBox.setChecked(True)
        self.faceCheckBox.stateChanged.connect(self.toggle_face_detection)
        self.handCheckBox = QCheckBox('Hands Detection')
        self.handCheckBox.setChecked(True)
        self.handCheckBox.stateChanged.connect(self.toggle_hand_detection)
        
        # create checkboxes for each command (index tip touch, pinch)
        self.indexTipTouchCheckBox = QCheckBox('Index Tip Touch Command')
        self.indexTipTouchCheckBox.setChecked(False)
        self.indexTipTouchCheckBox.stateChanged.connect(self.toggle_index_tip_touch_command)
        self.pinchCheckBox = QCheckBox('Pinch Command')
        self.pinchCheckBox.setChecked(False)
        self.pinchCheckBox.stateChanged.connect(self.toggle_pinch_command)

        # create slider for face model confidence
        self.faceModelConfidenceLabel = QLabel('Face Model Min. Confidence')
        self.faceModelConfidenceSlider = QSlider(Qt.Horizontal)
        self.faceModelConfidenceSlider.setFixedWidth(150)
        self.faceModelConfidenceSlider.setMinimum(0)
        self.faceModelConfidenceSlider.setMaximum(100)
        self.faceModelConfidenceSlider.setValue(50)
        self.faceModelConfidenceSlider.setTickPosition(QSlider.TicksBelow)
        self.faceModelConfidenceSlider.setTickInterval(10)
        self.faceModelConfidenceSlider.setSingleStep(10)
        self.faceModelConfidenceSlider.setPageStep(10)
        self.faceModelConfidenceSlider.setToolTip('Face Model Min. Confidence')
        self.faceModelConfidenceSlider.valueChanged.connect(self.set_face_model_confidence)
        self.faceModelConfidenceValueLabel = QLabel('0.5')

        # create slider for hand model confidence
        self.handModelConfidenceLabel = QLabel('Hand Model Min. Confidence')
        self.handModelConfidenceSlider = QSlider(Qt.Horizontal)
        self.handModelConfidenceSlider.setFixedWidth(150)
        self.handModelConfidenceSlider.setMinimum(0)
        self.handModelConfidenceSlider.setMaximum(100)
        self.handModelConfidenceSlider.setValue(50)
        self.handModelConfidenceSlider.setTickPosition(QSlider.TicksBelow)
        self.handModelConfidenceSlider.setTickInterval(10)
        self.handModelConfidenceSlider.setSingleStep(10)
        self.handModelConfidenceSlider.setPageStep(10)
        self.handModelConfidenceSlider.setToolTip('Hand Model Min. Confidence')
        self.handModelConfidenceSlider.valueChanged.connect(self.set_hand_model_confidence)
        self.handModelConfidenceValueLabel = QLabel('0.5')

        # create tab widget with two tabs for detections and parameters
        tabs = QTabWidget()
        tabs.addTab(self.detections_tab_UI(), 'Detections')
        tabs.addTab(self.parameters_tab_UI(), 'Parameters')

        # create layouts and add widgets/layouts to them
        vbox_main = QVBoxLayout()
        vbox_top = QVBoxLayout()
        vbox_bottom = QGridLayout()

        vbox_top.addWidget(self.image_label)
        vbox_top.addWidget(self.textLabel)
        vbox_top.addWidget(self.camButton)
        vbox_bottom.addWidget(tabs)

        vbox_main.addLayout(vbox_top)
        vbox_main.addLayout(vbox_bottom)

        # set the vbox layout as the widgets layout
        self.setLayout(vbox_main)

        # create the video capture thread
        self.thread = VideoThread()

        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)

    def detections_tab_UI(self):
        """It creates a tab in a tab widget, and adds the checkboxes to it."""
        detectionsTab = QWidget()
        detectionsTabLayout = QVBoxLayout()
        detectionsTabLayout.addWidget(self.faceCheckBox)
        detectionsTabLayout.addWidget(self.handCheckBox)
        detectionsTabLayout.addWidget(self.indexTipTouchCheckBox)
        detectionsTabLayout.addWidget(self.pinchCheckBox)
        detectionsTab.setLayout(detectionsTabLayout)
        return detectionsTab

    def parameters_tab_UI(self):
        """It creates a tab in a tab widget, and adds the sliders to it."""
        parametersTab = QWidget()
        parametersTabLayout = QVBoxLayout()
        parametersTabLayout.addWidget(self.faceModelConfidenceLabel)
        parametersTabLayout.addWidget(self.faceModelConfidenceSlider)
        parametersTabLayout.addWidget(self.faceModelConfidenceValueLabel)
        parametersTabLayout.addWidget(self.handModelConfidenceLabel)
        parametersTabLayout.addWidget(self.handModelConfidenceSlider)
        parametersTabLayout.addWidget(self.handModelConfidenceValueLabel)
        parametersTab.setLayout(parametersTabLayout)
        return parametersTab

    def set_face_model_confidence(self, value):
        """Scales the slider value before updating the model and setting the label"""
        scaledValue = float(value) / 100
        self.face.set_model_confidence(scaledValue)
        self.faceModelConfidenceValueLabel.setText(str(scaledValue))

    def set_hand_model_confidence(self, value):
        """Scales the slider value before updating the model and setting the label"""
        scaledValue = float(value) / 100
        self.hands.set_model_confidence(scaledValue)
        self.handModelConfidenceValueLabel.setText(str(scaledValue))

    def toggle_hand_detection(self):
        self.DETECT_HANDS = not self.DETECT_HANDS

    def toggle_face_detection(self):
        self.DETECT_FACE = not self.DETECT_FACE
    
    def toggle_index_tip_touch_command(self):
        self.hands.INDEX_TOUCH_COMMAND_ON = not self.hands.INDEX_TOUCH_COMMAND_ON

    def toggle_pinch_command(self):
        self.hands.PINCH_COMMAND_ON = not self.hands.PINCH_COMMAND_ON

    def toggle_webcam(self):
        """Toggles the webcam on or off"""
        if self.thread._run_flag == False:
            # Set the flag to true and start the thread
            self.thread._run_flag = True
            self.thread.start()
            self.camButton.setText('Stop Webcam')
        else:
            # Call stop() which sets flag to false and stops thread
            self.thread.stop()
            self.camButton.setText('Start Webcam')

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def process_detections(self, image):
        """Processes the image and detects hands and faces depending on the checkboxes"""
        if self.DETECT_HANDS:
            self.indexTipTouchCheckBox.setEnabled(True)
            self.pinchCheckBox.setEnabled(True)
            self.hands.load_image(image)
            hands_data, hands_type = self.hands.get_hand_data()

            # Do things with the hands (command checking, etc)
            for hand_data, hand_type in zip(hands_data, hands_type):
                if self.hands.INDEX_TOUCH_COMMAND_ON:
                    self.hands.index_tip_touch_check()
                if self.hands.PINCH_COMMAND_ON:
                    self.hands.pinch_check()
                if self.hands.BOUNDING_BOX_ON:
                    self.hands.bounding_box(hand_type)
                self.hands.update_landmark_positions(hand_data, hand_type)
        else:
            self.indexTipTouchCheckBox.setEnabled(False)
            self.pinchCheckBox.setEnabled(False)

        if self.DETECT_FACE:
            self.face.load_image(image)
            self.face.get_face_data()

    
if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())