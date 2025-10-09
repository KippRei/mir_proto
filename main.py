import sys
import functools # might need for partial
from new_audio_manager import AudioPlayer
from midi_manager import MIDIManager
from midi_controller import MIDIController
from zmkr_audio_processing import audio_preprocessor as zmkr
import qt_gui
from PyQt6.QtWidgets import QApplication

app = QApplication(sys.argv)
audio_manager = AudioPlayer()
audio_preprocessor = zmkr.AudioPreprocessor()
midi_manager = MIDIManager()
gui_window = qt_gui.QtGui(audio_manager, midi_manager, audio_preprocessor)
midi_controller = MIDIController(audio_manager, midi_manager)

midi_controller.quit_signal.connect(app.quit)
midi_controller.change_vol_signal.connect(gui_window.set_vol_slider)
midi_controller.change_pad_color_signal.connect(gui_window.set_button_color)

gui_window.show()
sys.exit(app.exec())