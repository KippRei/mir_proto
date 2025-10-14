import mido
from queue import Queue
import threading
import time

#Color values
RED = [127, 0, 0]
GREEN = [0, 127, 0]
BLUE = [0, 0, 127]
PURPLE = [127, 0, 127]
YELLOW = [127, 127, 0]
CYAN = [0, 127, 127]
ORANGE = [127, 64, 0]
LIME_GREEN = [64, 127, 0]
AQUA = [0, 127, 64]
HOT_PINK = [127, 0, 64]
LIGHT_BLUE = [0, 64, 127]
OLIVE = [64, 64, 0]
TEAL = [0, 64, 64]

PAD_OFF_COLOR = PURPLE
PAD_ON_COLOR = RED

class MIDIManager():
    def __init__(self):
        self.ports_open = False
        self.pad_color_map = {i: PAD_OFF_COLOR for i in range(36, 52)} # Creates map that holds pad/note # as key and [r,g,b] as value
        self.in_port = self.out_port = None
        self.midi_msg_queue = Queue() # queue to hold midi messages
        self.__open_ports('ATOM 0', 'ATOM 1')
        if self.ports_open:
            self.__set_midi_mode()
            self.set_pad_colors()
            # Start midi listener in separate thread
            msg_thread = threading.Thread(target=self.__listener, daemon=True)
            msg_thread.start()     

    # Opens MIDI ports
    def __open_ports(self, in_port_name, out_port_name):
        try:
            self.in_port = mido.open_input(in_port_name)
            self.out_port = mido.open_output(out_port_name)
            self.ports_open = True
        except:
            print("Ports not found")


    # Sets MIDI mode of physical controller (needs to be in native mode to control pad colors)
    def __set_midi_mode(self):
        self.out_port.send(mido.Message('note_off', channel=15, note=0, velocity=127))

    # MIDI listener for MIDI input (adds messages to midi message queue)
    def __listener(self):
        while True:
            for msg in self.in_port.iter_pending():
                self.midi_msg_queue.put(msg)
            time.sleep(0.015)

    # Sets the physical MIDI controller pad colors
    # TODO: Right now this iterates over every button every time it is updated
    # TODO (cont): Ideally it would only update the button that is pressed (and its column)
    def set_pad_colors(self):
        for k, v in self.pad_color_map.items():
            # Set channel 0 to 127 for light on or 0 for light off
            self.out_port.send(mido.Message('note_on', channel=0, note=k, velocity=0))
            self.out_port.send(mido.Message('note_on', channel=1, note=k, velocity=v[0]))
            self.out_port.send(mido.Message('note_on', channel=2, note=k, velocity=v[1]))
            self.out_port.send(mido.Message('note_on', channel=3, note=k, velocity=v[2]))

    def get_messages(self):
        while not self.midi_msg_queue.empty():
            yield self.midi_msg_queue.get()

    def get_pad_color_map(self):
        return self.pad_color_map

    def get_pad_on_off_color(self, val):
        return PAD_OFF_COLOR if val == 'off' else PAD_ON_COLOR