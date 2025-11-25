from PyQt6.QtWidgets import QSlider, QWidget, QGridLayout, QListWidget, QFileDialog, QPushButton, QLabel, QLCDNumber
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QIcon, QFont, QFontDatabase
from zqt_custom_widgets import SquareButton, PreprocessButton,SongListItem, SongList

class QtGui(QWidget):
    PLAY_BTN_PLAYING_IMG = './images/g_playing_edit.png'
    PLAY_BTN_STOPPED_IMG = './images/g_stopped_edit.png'

    def __init__(self, audio_manager, midi_manager, audio_preprocessor):
        super().__init__()
        self.audio_manager = audio_manager
        self.midi_manager = midi_manager
        self.audio_preprocessor = audio_preprocessor
        self.color_map = self.midi_manager.get_pad_color_map()
        self.audio_preprocessor.new_audio_preprocessed.connect(lambda: self.show_loading_bar(False))
        self.audio_preprocessor.new_audio_preprocessed.connect(lambda: self.update_song_list())
        font = QFontDatabase.addApplicationFont('./fonts/ZTNature-Medium.ttf')
        font_family = QFontDatabase.applicationFontFamilies(font)
        self.setFont(QFont(font_family[0], 10))

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
        self.setWindowTitle("Inside Out")
        self.setMinimumSize(800, 600)
        self.setWindowIcon(QIcon('./images/icon.ico'))

        self.__init_ui()

    def __init_ui(self):
        self.layout = QGridLayout()

        # Open/Drag and drop preprocess box
        self.preprocess_btn = PreprocessButton('Select file for preprocessing', self.audio_preprocessor)
        self.preprocess_btn.clicked.connect(self.open_file_to_preprocess)
        self.preprocess_btn.start_preprocessing.connect(lambda: self.show_loading_bar(True))
        self.layout.addWidget(self.preprocess_btn, 0, 0, alignment=Qt.AlignmentFlag.AlignVCenter)

        # List of songs/stems
        self.update_song_list()

        self.drum_vol = QSlider()
        self.bass_vol = QSlider()
        self.melody_vol = QSlider()
        self.vocal_vol = QSlider()
        
        sliders = {'drum': self.drum_vol, 'bass': self.bass_vol, 'melody': self.melody_vol, 'vocal': self.vocal_vol}
        for name, slider in sliders.items():
            slider.setFixedWidth(70)
            slider.setRange(0, 100)
            slider.setSingleStep(1)
            # slider.setFixedSize(20, 200)
            slider.setStyleSheet(self.stylesheet())
            # slider.setStyleSheet('QSlider::groove:vertical {background: purple}')
            self.set_vol_slider(name)

        self.layout.addWidget(self.drum_vol, 0, 1, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.layout.addWidget(self.bass_vol, 0, 2, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.layout.addWidget(self.melody_vol, 0, 3, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.layout.addWidget(self.vocal_vol, 0, 4, alignment=Qt.AlignmentFlag.AlignHCenter)

        for idx, key in enumerate(self.drum_buttons.keys()):
            self.drum_buttons[key] = SquareButton('Drums', key - 36, self.audio_manager)
            self.layout.addWidget(self.drum_buttons[key], idx + 1, 1, alignment=Qt.AlignmentFlag.AlignHCenter)

        for idx, key in enumerate(self.bass_buttons.keys()):
            self.bass_buttons[key] = SquareButton('Bass', key - 36, self.audio_manager)
            self.layout.addWidget(self.bass_buttons[key], idx + 1, 2, alignment=Qt.AlignmentFlag.AlignHCenter)

        for idx, key in enumerate(self.melody_buttons.keys()):
            self.melody_buttons[key] = SquareButton('Melody', key - 36, self.audio_manager)
            self.layout.addWidget(self.melody_buttons[key], idx + 1, 3, alignment=Qt.AlignmentFlag.AlignHCenter)

        for idx, key in enumerate(self.vocal_buttons.keys()):
            self.vocal_buttons[key] = SquareButton('Vocal', key - 36, self.audio_manager)
            self.layout.addWidget(self.vocal_buttons[key], idx + 1, 4, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.set_button_color()

        self.play_btn_img = QLabel(self)
        pic = QPixmap(self.PLAY_BTN_STOPPED_IMG)
        self.play_btn_img.setPixmap(pic)
        self.play_btn_img.setFixedHeight(90)
        self.play_btn_img.setFixedWidth(180)
        self.play_btn_img.setScaledContents(True)
        self.layout.addWidget(self.play_btn_img, 4, 0)

        self.tempo_img = QLCDNumber(self)
        self.tempo_img.setDigitCount(3)
        self.tempo_img.display(124)
        self.tempo_img.setFixedHeight(120)
        self.tempo_img.setFixedWidth(180)
        self.layout.addWidget(self.tempo_img, 3, 0)

        self.layout.setColumnStretch(0, 1)
        self.layout.setColumnStretch(1, 1)
        self.layout.setColumnStretch(2, 1)
        self.layout.setColumnStretch(3, 1)
        self.layout.setColumnStretch(4, 1)
        self.layout.setRowStretch(0, 1)
        self.layout.setRowStretch(1, 2)
        self.layout.setRowStretch(2, 2)
        self.layout.setRowStretch(3, 2)
        self.layout.setRowStretch(4, 2)

        self.setLayout(self.layout)

    def stylesheet(self):
        return """
            QSlider::groove:vertical {
                background: red;
                position: absolute; /* absolutely position 4px from the left and right of the widget. setting margins on the widget should work too... */
                left: 30px; right: 30px;
                border-radius: 10px;
            }

            QSlider::handle:vertical {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #000000, stop:1 purple);
                border: 1px solid purple;
                height: 8px;
                width: 18px;
                margin: 0 -100px; /* handle is placed by default on the contents rect of the groove. Expand outside the groove */
                border-radius: 10px;
            }

            QSlider::add-page:vertical {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #000000, stop:1 purple);;
            }

            QSlider::sub-page:vertical {
                background: black;
            }

            QPushButton {
                border: 2px solid #8f8f91;
                border-radius: 6px;
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                                stop: 0 #f6f7fa, stop: 1 #dadbde);
                min-width: 80px;
            }

            QPushButton:pressed {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                                stop: 0 #dadbde, stop: 1 #f6f7fa);
            }

            QPushButton:flat {
                border: none; /* no border for a flat push button */
            }

            QPushButton:default {
                border-color: navy; /* make the default button prominent */
            }
        """

    # Changes tempo
    def set_tempo(self, amt):
        curr_tempo = int(self.tempo_img.value())
        self.tempo_img.display(curr_tempo + amt)

    def set_play_btn(self, on_off):
        if on_off:
            self.play_btn_img.setPixmap(QPixmap(self.PLAY_BTN_PLAYING_IMG))
            

        else:
            self.play_btn_img.setPixmap(QPixmap(self.PLAY_BTN_STOPPED_IMG)) 
        

    # Updates list of songs
    def update_song_list(self):
        self.song_list = SongList(self)
        self.song_list.setDragEnabled(True)
        self.song_arr = []
        # Add name of each stem to song list
        for song_name in self.audio_manager.get_song_list():
            self.song_arr.append(SongListItem(song_name, self.song_list))
        self.layout.addWidget(self.song_list, 1, 0, 2, 1, alignment=Qt.AlignmentFlag.AlignLeft)

    # Open file to preprocess
    def open_file_to_preprocess(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)
        
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            for f in selected_files:
                self.show_loading_bar(True)
                self.audio_preprocessor.process_audio(f)

    # Displays loading bar on preprocess button when song is being processed
    def show_loading_bar(self, show_flag):
        if show_flag:
            self.preprocess_btn.setEnabled(False)
            self.preprocess_btn.setText('Preprocessing song(s)...')
        else:
            self.preprocess_btn.setEnabled(True)
            self.preprocess_btn.setText('Select file for preprocessing')


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