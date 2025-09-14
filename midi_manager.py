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
        self.pad_color_map = {i: PAD_OFF_COLOR for i in range(36, 52)} # Creates map that holds pad/note # as key and [r,g,b] as value
        self.in_port = self.out_port = None
        self.midi_msg_queue = Queue() # queue to hold midi messages
        self.__open_ports('ATOM 0', 'ATOM 1')
        self.__set_midi_mode()
        self.__set_pad_colors()
        # Start midi listener in separate thread
        msg_thread = threading.Thread(target=self.__listener, daemon=True)
        msg_thread.start()     

    def __open_ports(self, in_port_name, out_port_name):
        self.in_port = mido.open_input(in_port_name)
        self.out_port = mido.open_output(out_port_name)

    def __set_midi_mode(self):
        self.out_port.send(mido.Message('note_off', channel=15, note=0, velocity=127))

    def __set_pad_colors(self):
        for k, v in self.pad_color_map.items():
            self.out_port.send(mido.Message('note_on', channel=0, note=k, velocity=127))
            self.out_port.send(mido.Message('note_on', channel=1, note=k, velocity=v[0]))
            self.out_port.send(mido.Message('note_on', channel=2, note=k, velocity=v[1]))
            self.out_port.send(mido.Message('note_on', channel=3, note=k, velocity=v[2]))

    def __listener(self):
        while True:
            for msg in self.in_port.iter_pending():
                self.midi_msg_queue.put(msg)
            time.sleep(0.015)

    def get_messages(self):
        while not self.midi_msg_queue.empty():
            yield self.midi_msg_queue.get()

    def change_pad_color(self, note_num):
        col_mod = (note_num - 36) % 4
        for k in self.pad_color_map:
            if k == note_num:
                self.pad_color_map[k] = PAD_ON_COLOR if self.pad_color_map[k] == PAD_OFF_COLOR else PAD_OFF_COLOR
            elif k % 4 == col_mod:
                self.pad_color_map[k] = PAD_OFF_COLOR # note off color = purple

        self.__set_pad_colors()