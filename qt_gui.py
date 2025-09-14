from PyQt6.QtWidgets import QSlider, QWidget, QGridLayout, QPushButton
from PyQt6.QtCore import Qt

class QtGui(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("InsideOut")
        self.setGeometry(0, 0, 800, 600)
        self.__init_ui()

    def __init_ui(self):
        layout = QGridLayout()

        self.drum_vol = QSlider()
        self.bass_vol = QSlider()
        self.melody_vol = QSlider()
        self.vocal_vol = QSlider()

        self.drum_vol.setRange(0, 100)
        self.drum_vol.setValue(100)
        self.drum_vol.setSingleStep(1)
        self.bass_vol.setRange(0, 100)
        self.bass_vol.setValue(100)
        self.bass_vol.setSingleStep(1)
        self.melody_vol.setRange(0, 100)
        self.melody_vol.setValue(100)
        self.melody_vol.setSingleStep(1)
        self.vocal_vol.setRange(0, 100)
        self.vocal_vol.setValue(100)
        self.vocal_vol.setSingleStep(1)

        self.drum_btn_1 = QPushButton('Drums (on)')
        self.drum_btn_1.setFixedSize(100, 100)
        self.drum_btn_2 = QPushButton('Drums (on)')
        self.drum_btn_2.setFixedSize(100, 100)
        self.drum_btn_3 = QPushButton('Drums (on)')
        self.drum_btn_3.setFixedSize(100, 100)
        self.drum_btn_4 = QPushButton('Drums (on)')
        self.drum_btn_4.setFixedSize(100, 100)

        self.vocal_btn_1 = QPushButton('Vocals (on)')
        self.vocal_btn_1.setFixedSize(100, 100)
        self.vocal_btn_2 = QPushButton('Vocals (on)')
        self.vocal_btn_2.setFixedSize(100, 100)
        self.vocal_btn_3 = QPushButton('Vocals (on)')
        self.vocal_btn_3.setFixedSize(100, 100)
        self.vocal_btn_4 = QPushButton('Vocals (on)')
        self.vocal_btn_4.setFixedSize(100, 100)

        self.melody_btn_1 = QPushButton('Melody (on)')
        self.melody_btn_1.setFixedSize(100, 100)
        self.melody_btn_2 = QPushButton('Melody (on)')
        self.melody_btn_2.setFixedSize(100, 100)
        self.melody_btn_3 = QPushButton('Melody (on)')
        self.melody_btn_3.setFixedSize(100, 100)
        self.melody_btn_4 = QPushButton('Melody (on)')
        self.melody_btn_4.setFixedSize(100, 100)

        self.bass_btn_1 = QPushButton('Bass (on)')
        self.bass_btn_1.setFixedSize(100, 100)
        self.bass_btn_2 = QPushButton('Bass (on)')
        self.bass_btn_2.setFixedSize(100, 100)
        self.bass_btn_3 = QPushButton('Bass (on)')
        self.bass_btn_3.setFixedSize(100, 100)
        self.bass_btn_4 = QPushButton('Bass (on)')
        self.bass_btn_4.setFixedSize(100, 100)

        layout.addWidget(self.drum_vol, 0, 0, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.bass_vol, 0, 1, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.melody_vol, 0, 2, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.vocal_vol, 0, 3, alignment=Qt.AlignmentFlag.AlignHCenter)

        layout.addWidget(self.drum_btn_1, 1, 0)
        layout.addWidget(self.drum_btn_2, 2, 0)
        layout.addWidget(self.drum_btn_3, 3, 0)
        layout.addWidget(self.drum_btn_4, 4, 0)

        layout.addWidget(self.bass_btn_1, 1, 1)
        layout.addWidget(self.bass_btn_2, 2, 1)
        layout.addWidget(self.bass_btn_3, 3, 1)
        layout.addWidget(self.bass_btn_4, 4, 1)

        layout.addWidget(self.melody_btn_1, 1, 2)
        layout.addWidget(self.melody_btn_2, 2, 2)
        layout.addWidget(self.melody_btn_3, 3, 2)
        layout.addWidget(self.melody_btn_4, 4, 2)

        layout.addWidget(self.vocal_btn_1, 1, 3)
        layout.addWidget(self.vocal_btn_2, 2, 3)
        layout.addWidget(self.vocal_btn_3, 3, 3)
        layout.addWidget(self.vocal_btn_4, 4, 3)

        self.setLayout(layout)