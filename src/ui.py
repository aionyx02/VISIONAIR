from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QSlider
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisionAir")

        self.layout = QVBoxLayout()

        self.video_label = QLabel()
        self.layout.addWidget(self.video_label)

        self.label = QLabel("Hand Gesture: ")
        self.layout.addWidget(self.label)

        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setRange(1, 20)
        self.sensitivity_slider.setValue(3)    # 設定初始值
        self.layout.addWidget(self.sensitivity_slider)

        self.central_widget = QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

    def update_frame(self, frame):
        """更新視訊畫面"""
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        p = convert_to_Qt_format.scaled(1024, 960, Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(p))

    def update_gesture(self, gesture):
        """更新手勢標籤"""
        self.label.setText(f"Hand Gesture: {gesture}")

    def get_sensitivity(self):
        """獲取靈敏度滑桿的值"""
        return self.sensitivity_slider.value()