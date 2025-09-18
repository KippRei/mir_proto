from PyQt6.QtCore import QTimer, QObject, pyqtSignal

class MIDIController(QObject):
    quit_signal = pyqtSignal()
    change_vol_signal = pyqtSignal(str)
    change_pad_color_signal = pyqtSignal()

    def __init__(self, audio_manager, midi_manager):
        super().__init__()
        self.midi_manager = midi_manager
        self.audio_manager = audio_manager
        self.channel_map = self.audio_manager.get_channel_map()
        self.pad_color_map = self.midi_manager.get_pad_color_map()
        self.midi_timer = QTimer()
        self.midi_timer.setInterval(10)
        self.midi_timer.timeout.connect(self.process_midi_messages)
        self.midi_timer.start()
        self.change_pad_color() # TODO: just calling this because tracks autoplay right now, remove after testing

    # Controls what happens when MIDI message is received
    def process_midi_messages(self):
        for msg in self.midi_manager.get_messages():
            print(msg)
            match msg.type:
                case 'note_on':
                    match msg.note:
                        case 36:
                            self.audio_manager.play_channel(0)
                        case 37:
                            self.audio_manager.play_channel(1)
                        case 38:
                            self.audio_manager.play_channel(2)
                        case 39:
                            self.audio_manager.play_channel(3)
                        case 40:
                            self.audio_manager.play_channel(4)
                        case 41:
                            self.audio_manager.play_channel(5)
                        case 42:
                            self.audio_manager.play_channel(6)
                        case 43:
                            self.audio_manager.play_channel(7)
                        case 44:
                            self.audio_manager.play_channel(8)
                        case 45:
                            self.audio_manager.play_channel(9)
                        case 46:
                            self.audio_manager.play_channel(10)
                        case 47:
                            self.audio_manager.play_channel(11)
                        case 48:
                            self.audio_manager.play_channel(12)
                        case 49:
                            self.audio_manager.play_channel(13)
                        case 50:
                            self.audio_manager.play_channel(14)
                        case 51:
                            self.audio_manager.play_channel(15)

                    self.change_pad_color()

                case 'control_change':
                    match msg.control:
                        case 14:
                            if msg.value <= 64:
                                self.audio_manager.adj_track_vol('drum', 0.05)
                            elif msg.value >= 65:
                                self.audio_manager.adj_track_vol('drum', -0.05)
                            self.change_vol_signal.emit('drum')
                        case 15:
                            if msg.value == 1:
                                self.audio_manager.adj_track_vol('bass', 0.05)
                            elif msg.value == 65:
                                self.audio_manager.adj_track_vol('bass', -0.05)
                            self.change_vol_signal.emit('bass')
                        case 16:
                            if msg.value == 1:
                                self.audio_manager.adj_track_vol('melody', 0.05)
                            elif msg.value == 65:
                                self.audio_manager.adj_track_vol('melody', -0.05)
                            self.change_vol_signal.emit('melody')
                        case 17:
                            if msg.value == 1:
                                self.audio_manager.adj_track_vol('vocal', 0.05)
                            elif msg.value == 65:
                                self.audio_manager.adj_track_vol('vocal', -0.05)
                            self.change_vol_signal.emit('vocal')
                        # Stop button
                        case 111:
                            if msg.value == 127:
                                self.audio_manager.hit_stop()
                        case 109:
                            if msg.value == 127:
                                self.audio_manager.hit_play()

    # Logic for changing pad color map values to 'ON' for button pressed and changes color of all other buttons in column to 'OFF'
    def change_pad_color(self):
        for k in self.channel_map.keys():
            self.pad_color_map[k+36] = self.midi_manager.get_pad_on_off_color('on') if self.channel_map[k]['is_playing'] is True else self.midi_manager.get_pad_on_off_color('off')
    
        # Tell midi manager to update colors
        self.midi_manager.set_pad_colors()
        # Emit signal to tell GUI to update
        self.change_pad_color_signal.emit()