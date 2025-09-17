import tkinter as tk
from functools import partial
import mido
from threading import Thread
from queue import Queue
import audio_manager as ac

# from gemini -> connect to midi controller
input_midi_port_name = 'ATOM 0' 
output_midi_port_name = 'ATOM 1' 

midi_msg_queue = Queue()

def midi_listener():
    try:
        print(f"Opening port: {input_midi_port_name}...")
        with mido.open_input(input_midi_port_name) as port:
            for msg in port:
                midi_msg_queue.put(msg)
                # Check if the message is a 'note_on' message (i.e., a pad was pressed)
                if msg.type == 'note_on':
                    print(f"Pad pressed! Note: {msg.note}, Velocity: {msg.velocity}")
                
                # Check for other message types, like 'control_change'
                elif msg.type == 'control_change':
                    print(f"Control changed! Controller: {msg.control}, Value: {msg.value}")
                
                # You can handle other message types here as needed
                else:
                    print(f"Received message: {msg}")

    except mido.PortNotOpenError:
        print(f"Error: Could not open port '{input_midi_port_name}'. Make sure your device is connected and recognized.")
        print("Available ports:")
        print(mido.get_input_names())
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        print("Port closed.")


class TkApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.s1 = self.s2 = self.s3 = self.s4 = self.s5 = None
        self.button1 = self.button2 = self.button3 = self.button4 = self.button5 = None
        self.drum1_on = self.melody1_on = self.vocal1_on = False
        self.m_thread = Thread(target=self.init_midi_listener, daemon=True)
        self.m_thread.start()
        self.init_ui()

        self.after(15, self.init_midi_listener)

    def init_ui(self):
        self.title('InsideOut')
        self.geometry('500x400')
        self.s1 = tk.Scale(self, from_=1, to=0, resolution=0.01)
        self.s2 = tk.Scale(self, from_=1, to=0, resolution=0.01)
        self.s3 = tk.Scale(self, from_=1, to=0, resolution=0.01)
        self.s4 = tk.Scale(self, from_=1, to=0, resolution=0.01)
        self.s1.set(1)
        self.s2.set(1)
        self.s3.set(1)
        self.s4.set(1)
        self.button1 = tk.Button(self, text='Drums (on)', width=25, pady=20, command=self.play_drums)
        self.button2 = tk.Button(self, text='Melody 1', width=25, pady=20, command=partial(self.play_melody, 1))
        self.button3 = tk.Button(self, text='Vocals (on)', width=25, pady=20, command=self.play_vocals)
        self.button4 = tk.Button(self, text='Drum (hit)', width=25, pady=20)
        self.button5 = tk.Button(self, text='Melody 2', width=25, pady=20, command=partial(self.play_melody, 2))

        self.button1.grid(row=0, column=0)
        self.button2.grid(row=1, column=0)
        self.button3.grid(row=2, column=0)
        self.button4.grid(row=3, column=0)
        self.button5.grid(row=1, column=2)

        self.s1.grid(row=0, column=1)
        self.s2.grid(row=1, column=1)
        self.s3.grid(row=2, column=1)
        self.s4.grid(row=3, column=1)

    def init_midi_listener(self):
        while not midi_msg_queue.empty():
            midi_msg = midi_msg_queue.get()
            if midi_msg.type == 'note_on':
                self.process_midi_input(midi_msg.note)
            elif midi_msg.type == 'note_off':
                self.drum1_on = False
                self.melody1_on = False
                self.vocal1_on = False
            elif midi_msg.type == 'control_change':
                self.process_midi_control_change(midi_msg)
        
        self.after(10, self.init_midi_listener)

    def process_midi_control_change(self, msg):
        if msg.control == 14:
            self.update_vol('drums', msg.value)
        if msg.control == 16:
            self.update_vol('melody', msg.value)
        if msg.control == 17:
            self.update_vol('vocals', msg.value)

    def process_midi_input(self, note_val):
        if note_val == 36 and not self.drum1_on:
            self.play_drums()
            self.drum1_on = True
        if note_val == 37 and not self.drum1_on:
            #bass part
            print('bass pressed')
        if note_val == 38 and not self.melody1_on:
            self.play_melody(1)
            self.melody1_on = True
        if note_val == 42 and not self.melody1_on:
            self.play_melody(2)
            self.melody1_on = True
        if note_val == 39 and not self.vocal1_on:
            self.play_vocals()
            self.vocal1_on = True



# button4.bind('<ButtonPress-1>', start_drum_hit)
# button4.bind('<ButtonRelease-1>', stop_drum_hit)
# button4.pack()





if __name__ == '__main__':

    # Thanks to https://github.com/EMATech/AtomCtrl
    # TODO: create color control script
    # TODO: fix volume knobs, in native mode they give value of 1 if turned left and 65 if turned right so need to add/subtract from volume (can't rely on knob value)
    # Pad color control numbers (CC numbers)
    CC_PAD_DRUMS = 36
    CC_PAD_STRINGS = 37
    CC_PAD_VOCALS = 38

    # Color values
    RED = [127, 0, 0]
    GREEN = [0, 127, 0]
    BLUE = [0, 0, 127]

    try:
        out_port = mido.open_output(output_midi_port_name)
    except Exception as e:
        print(f"Error opening MIDI output port: {e}")
        # sys.exit(1)

    # Put MIDI controller in native mode
    out_port.send(mido.Message('note_off', channel=15, note=0, velocity=127))
    # Initialize the button colors
    try:
        '''channel 0 '''
        out_port.send(mido.Message('note_on', channel=0, note=36, velocity=127))
        out_port.send(mido.Message('note_on', channel=1, note=36, velocity=GREEN[0]))
        out_port.send(mido.Message('note_on', channel=2, note=36, velocity=GREEN[1]))
        out_port.send(mido.Message('note_on', channel=3, note=36, velocity=GREEN[2]))
    except Exception as e:
        print(f"Error initializing pad colors: {e}")


    t = Thread(target=midi_listener, daemon=True)
    t.start()
    audio_control = ac.AudioController()
    window = TkApp()
    window.mainloop()
