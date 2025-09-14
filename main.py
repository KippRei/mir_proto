import sys
from audio_manager import AudioPlayer
from midi_manager import MIDIManager
from midi_controller import MIDIController
import qt_gui
from PyQt6.QtWidgets import QApplication

app = QApplication(sys.argv)
music = AudioPlayer()
midi_manager = MIDIManager()
gui_window = qt_gui.QtGui()
midi_controller = MIDIController(midi_manager)

midi_controller.quit_signal.connect(app.quit)

gui_window.show()
sys.exit(app.exec())