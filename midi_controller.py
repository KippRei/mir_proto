from PyQt6.QtCore import QTimer, QObject, pyqtSignal

class MIDIController(QObject):
    quit_signal = pyqtSignal()
    change_vol_signal = pyqtSignal(str)
    change_pad_color_signal = pyqtSignal()

    def __init__(self, audio_manager, midi_manager):
        super().__init__()
        self.midi_manager = midi_manager
        self.audio_manager = audio_manager
        self.pad_color_map = self.midi_manager.get_pad_color_map()
        self.midi_timer = QTimer()
        self.midi_timer.setInterval(10)
        self.midi_timer.timeout.connect(self.process_midi_messages)
        self.midi_timer.start()

    # Controls what happens when MIDI message is received
    def process_midi_messages(self):
        for msg in self.midi_manager.get_messages():
            print(msg)
            match msg.type:
                case 'note_on':
                    self.change_pad_color(msg.note)
                case 'control_change':
                    match msg.control:
                        case 14:
                            if msg.value <= 64:
                                self.audio_manager.adj_track_vol('drum', 5)
                            elif msg.value >= 65:
                                self.audio_manager.adj_track_vol('drum', -5)
                            self.change_vol_signal.emit('drum')
                        case 15:
                            if msg.value == 1:
                                self.audio_manager.adj_track_vol('bass', 3)
                            elif msg.value == 65:
                                self.audio_manager.adj_track_vol('bass', -3)
                            self.change_vol_signal.emit('bass')
                        case 16:
                            if msg.value == 1:
                                self.audio_manager.adj_track_vol('melody', 3)
                            elif msg.value == 65:
                                self.audio_manager.adj_track_vol('melody', -3)
                            self.change_vol_signal.emit('melody')
                        case 17:
                            if msg.value == 1:
                                self.audio_manager.adj_track_vol('vocal', 3)
                            elif msg.value == 65:
                                self.audio_manager.adj_track_vol('vocal', -3)
                            self.change_vol_signal.emit('vocal')
                        # Stop button
                        case 111:
                            self.quit_signal.emit()

    # Logic for changing pad color map values to 'ON' for button pressed and changes color of all other buttons in column to 'OFF'
    def change_pad_color(self, note_num):
        col_mod = (note_num - 36) % 4
        for k in self.pad_color_map:
            if k == note_num:
                self.pad_color_map[k] = self.midi_manager.get_pad_on_off_color('on') if self.pad_color_map[k] == self.midi_manager.get_pad_on_off_color('off') else self.midi_manager.get_pad_on_off_color('off')
            elif k % 4 == col_mod:
                self.pad_color_map[k] = self.midi_manager.get_pad_on_off_color('off') # note off color = purple

        # Tell midi manager to update colors
        self.midi_manager.set_pad_colors()
        # Emit signal to tell GUI to update
        self.change_pad_color_signal.emit()