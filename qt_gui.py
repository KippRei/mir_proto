from PyQt6.QtWidgets import QSlider, QWidget, QGridLayout, QPushButton
from PyQt6.QtCore import Qt

class QtGui(QWidget):
    def __init__(self, audio_manager, midi_manager):
        super().__init__()
        self.audio_manager = audio_manager
        self.midi_manager = midi_manager
        self.color_map = self.midi_manager.get_pad_color_map()
        self.drum_buttons = {
            48: None,
            44: None,
            40: None,
            36: None
        }
        self.bass_buttons = {
            49: None,
            45: None,
            41: None,
            37: None
        }
        self.melody_buttons = {
            50: None,
            46: None,
            42: None,
            38: None
        }
        self.vocal_buttons = {
            51: None,
            47: None,
            43: None,
            39: None
        }
        self.buttons_arr = [self.drum_buttons, self.bass_buttons, self.melody_buttons, self.vocal_buttons]
        self.setWindowTitle("InsideOut")
        self.setGeometry(0, 0, 800, 600)
        self.__init_ui()

    def __init_ui(self):
        layout = QGridLayout()

        self.drum_vol = QSlider()
        self.bass_vol = QSlider()
        self.melody_vol = QSlider()
        self.vocal_vol = QSlider()
        sliders = {'drum': self.drum_vol, 'bass': self.bass_vol, 'melody': self.melody_vol, 'vocal': self.vocal_vol}
        for name, slider in sliders.items():
            slider.setRange(0, 100)
            slider.setSingleStep(1)
            # slider.setStyleSheet('QSlider::groove:vertical {background: purple}')
            self.set_vol_slider(name)

        layout.addWidget(self.drum_vol, 0, 0, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.bass_vol, 0, 1, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.melody_vol, 0, 2, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.vocal_vol, 0, 3, alignment=Qt.AlignmentFlag.AlignHCenter)


        for idx, key in enumerate(self.drum_buttons.keys()):
            self.drum_buttons[key] = QPushButton('Drums')
            self.drum_buttons[key].setFixedSize(100, 100)
            layout.addWidget(self.drum_buttons[key], idx + 1, 0)
        for idx, key in enumerate(self.bass_buttons.keys()):
            self.bass_buttons[key] = QPushButton('Bass')
            self.bass_buttons[key].setFixedSize(100, 100)
            layout.addWidget(self.bass_buttons[key], idx + 1, 1)
        for idx, key in enumerate(self.melody_buttons.keys()):
            self.melody_buttons[key] = QPushButton('Melody')
            self.melody_buttons[key].setFixedSize(100, 100)
            layout.addWidget(self.melody_buttons[key], idx + 1, 2)
        for idx, key in enumerate(self.vocal_buttons.keys()):
            self.vocal_buttons[key] = QPushButton('Vocal')
            self.vocal_buttons[key].setFixedSize(100, 100)
            layout.addWidget(self.vocal_buttons[key], idx + 1, 3)

        self.set_button_color()

        self.setLayout(layout)

    def set_vol_slider(self, name):
        match name:
            case 'drum':
                self.drum_vol.setValue(self.audio_manager.get_track_vol('drum'))
            case 'bass':
                self.bass_vol.setValue(self.audio_manager.get_track_vol('bass'))
            case 'melody':
                self.melody_vol.setValue(self.audio_manager.get_track_vol('melody'))
            case 'vocal':
                self.vocal_vol.setValue(self.audio_manager.get_track_vol('vocal'))

    def set_button_color(self):   
        for button_set in self.buttons_arr:
            for key in button_set.keys():
                button_set[key].setStyleSheet(f'background-color: rgb({self.color_map[key][0]}, {self.color_map[key][1]}, {self.color_map[key][2]})')