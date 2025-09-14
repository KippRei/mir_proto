from PyQt6.QtCore import QTimer, QObject, pyqtSignal

class MIDIController(QObject):
    quit_signal = pyqtSignal()

    def __init__(self, midi_manager, audio_manager):
        super().__init__()
        self.midi_manager = midi_manager
        self.audio_manager = audio_manager
        self.midi_timer = QTimer()
        self.midi_timer.setInterval(10)
        self.midi_timer.timeout.connect(self.process_midi_messages)
        self.midi_timer.start()

    def process_midi_messages(self):
        for msg in self.midi_manager.get_messages():
            print(msg)
            match msg.type:
                case 'note_on':
                    self.midi_manager.change_pad_color(msg.note)
                case 'control_change':
                    match msg.control:
                        case 14:
                            if msg.value == 1:
                                self.audio_manager.adj_track_vol('drum', 3)
                            elif msg.value == 65:
                                self.audio_manager.adj_track_vol('drum', -3)
                        case 15:
                            pass
                        case 16:
                            pass
                        case 17:
                            pass
                        case 111:
                            self.quit_signal.emit()