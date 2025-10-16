import sys
import functools # might need for partial
from rust_audio_manager import AudioPlayer
from midi_manager import MIDIManager
from midi_controller import MIDIController
from zmkr_audio_processing import audio_preprocessor as zmkr
import qt_gui
from PyQt6.QtWidgets import QApplication

# print('starting play')
# zmkr_audio_engine.play()
# print('play over')
app = QApplication(sys.argv)
audio_manager = AudioPlayer()
midi_manager = MIDIManager()
midi_controller = MIDIController(audio_manager, midi_manager)
audio_preprocessor = zmkr.AudioPreprocessor()
gui_window = qt_gui.QtGui(audio_manager, midi_manager, audio_preprocessor)

midi_controller.quit_signal.connect(app.quit)
midi_controller.change_vol_signal.connect(gui_window.set_vol_slider)
midi_controller.change_pad_color_signal.connect(gui_window.set_button_color)
audio_preprocessor.new_audio_preprocessed.connect(audio_manager.load_preprocessed_songs)
audio_preprocessor.new_audio_preprocessed.connect(gui_window.update_song_list)

gui_window.show()
sys.exit(app.exec())