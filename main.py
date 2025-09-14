from audio_manager import AudioPlayer
from midi_manager import MIDIManager
from midi_controller import MIDIController
import gui    

music = AudioPlayer()
midi_manager = MIDIManager()
gui_window = gui.TkApp()
midi_controller = MIDIController(midi_manager, gui_window)

midi_controller.process_midi_messages()
gui_window.mainloop()