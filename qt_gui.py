from PyQt6.QtWidgets import QSlider, QWidget, QGridLayout, QListWidget, QFileDialog
from PyQt6.QtCore import Qt
from qt_square_button import SquareButton
from qt_song_list_widget import SongListItem
import os

class QtGui(QWidget):
    def __init__(self, audio_manager, midi_manager):
        super().__init__()
        self.audio_manager = audio_manager
        self.midi_manager = midi_manager
        self.color_map = self.midi_manager.get_pad_color_map()

        # Button maps = {MIDI controller pad num: QPushButton}
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

        # Open/Drag and drop preprocess box
        self.preprocess_btn = SquareButton('Select file for preprocessing')
        self.preprocess_btn.setFixedSize(200, 200)
        self.preprocess_btn.clicked.connect(self.open_file_to_preprocess)
        layout.addWidget(self.preprocess_btn, 0, 0, alignment=Qt.AlignmentFlag.AlignVCenter)

        # List of songs/stems
        # TODO: Refactor this!
        self.song_list = QListWidget()
        self.song_list.setDragEnabled(True)
        self.song_arr = []
        song_dir = os.fsencode('./preprocessed_audio')
        for idx, file in enumerate(os.listdir(song_dir)):
            self.song_arr.append(SongListItem(os.fsdecode(file).split('.')[0], self.song_list))
        layout.addWidget(self.song_list, 1, 0, 4, 1, alignment=Qt.AlignmentFlag.AlignLeft)

        self.drum_vol = QSlider()
        self.bass_vol = QSlider()
        self.melody_vol = QSlider()
        self.vocal_vol = QSlider()
        sliders = {'drum': self.drum_vol, 'bass': self.bass_vol, 'melody': self.melody_vol, 'vocal': self.vocal_vol}
        for name, slider in sliders.items():
            slider.setRange(0, 100)
            slider.setSingleStep(1)
            slider.setFixedSize(20, 200)
            # slider.setStyleSheet('QSlider::groove:vertical {background: purple}')
            self.set_vol_slider(name)

        layout.addWidget(self.drum_vol, 0, 1, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.bass_vol, 0, 2, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.melody_vol, 0, 3, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.vocal_vol, 0, 4, alignment=Qt.AlignmentFlag.AlignHCenter)


        for idx, key in enumerate(self.drum_buttons.keys()):
            self.drum_buttons[key] = SquareButton('Drums')
            # self.drum_buttons[key].setFixedSize(100, 100)
            layout.addWidget(self.drum_buttons[key], idx + 1, 1)
        for idx, key in enumerate(self.bass_buttons.keys()):
            self.bass_buttons[key] = SquareButton('Bass')
            self.bass_buttons[key].setAcceptDrops(True)
            # self.bass_buttons[key].setFixedSize(100, 100)
            layout.addWidget(self.bass_buttons[key], idx + 1, 2)
        for idx, key in enumerate(self.melody_buttons.keys()):
            self.melody_buttons[key] = SquareButton('Melody')
            self.melody_buttons[key].setAcceptDrops(True)
            # self.melody_buttons[key].setFixedSize(100, 100)
            layout.addWidget(self.melody_buttons[key], idx + 1, 3)
        for idx, key in enumerate(self.vocal_buttons.keys()):
            self.vocal_buttons[key] = SquareButton('Vocal')
            self.vocal_buttons[key].setAcceptDrops(True)
            # self.vocal_buttons[key].setFixedSize(100, 100)
            layout.addWidget(self.vocal_buttons[key], idx + 1, 4)

        self.set_button_color()

        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        layout.setColumnStretch(3, 1)
        layout.setColumnStretch(4, 1)
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 2)
        layout.setRowStretch(2, 2)
        layout.setRowStretch(3, 2)
        layout.setRowStretch(4, 2)

        self.setLayout(layout)

    # Open file to preprocess
    def open_file_to_preprocess(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)
        
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            for f in selected_files:
                print(f)

    # Sets position of volume slider
    def set_vol_slider(self, name):
        vol = self.audio_manager.get_track_vol(name) * 100
        vol = int(vol)
        match name:
            case 'drum':
                self.drum_vol.setValue(vol)
            case 'bass':
                self.bass_vol.setValue(vol)
            case 'melody':
                self.melody_vol.setValue(vol)
            case 'vocal':
                self.vocal_vol.setValue(vol)

    # Sets GUI button colors
    # TODO: Right now this iterates over every button every time it is updated
    # TODO (cont): Ideally it would only update the button that is pressed (and its column)
    def set_button_color(self):   
        for button_set in self.buttons_arr:
            for key in button_set.keys():
                button_set[key].setStyleSheet(f'background-color: rgb({self.color_map[key][0]}, {self.color_map[key][1]}, {self.color_map[key][2]})')