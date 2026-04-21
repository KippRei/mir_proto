"""
Instantiates all necessary classes for program execution, starts listeners for midi_controller and audio_preprocessor, and starts PyQt event loop
"""

import sys
import functools # might need for partial
from rust_audio_manager import AudioPlayer
from midi_manager import MIDIManager
from midi_controller import MIDIController
from zmkr_audio_processing import audio_preprocessor as zmkr
import qt_gui
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont, QFontDatabase

# Initialize that PyQt application
app = QApplication(sys.argv)

# Instantiate audio player and MIDI manager
audio_manager = AudioPlayer()
midi_manager = MIDIManager()

# Initialize MIDI controller
midi_controller = MIDIController(audio_manager, midi_manager)

# Initialize the audio preprocessor
audio_preprocessor = zmkr.AudioPreprocessor()

# Create GUI window
gui_window = qt_gui.QtGui(audio_manager, midi_manager, audio_preprocessor, midi_controller)

# Set up listeners for various inputs
midi_controller.change_vol_signal.connect(gui_window.set_vol_slider)
midi_controller.change_pad_color_signal.connect(gui_window.set_button_color)
midi_controller.change_tempo_signal.connect(gui_window.set_tempo)
midi_controller.change_playing_signal.connect(gui_window.set_play_btn)

audio_preprocessor.new_audio_preprocessed.connect(audio_manager.load_preprocessed_songs)
audio_preprocessor.new_audio_preprocessed.connect(gui_window.update_song_list)

# Show GUI window
gui_window.show()

# Start main event loop
sys.exit(app.exec())