class MIDIController():
    def __init__(self, midi_manager, gui_window):
        self.midi_manager = midi_manager
        self.gui_window = gui_window

    def process_midi_messages(self):
        for msg in self.midi_manager.get_messages():
            match msg.type:
                case 'note_on':
                    self.midi_manager.change_pad_color(msg.note)
                case 'control_change':
                    match msg.control:
                        case 111:
                            self.gui_window.quit()
                            
        self.gui_window.after(15, self.process_midi_messages)

    