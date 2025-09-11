import tkinter as tk
import pygame
from functools import partial
import mido
from threading import Thread
import time

# from gemini -> connect to midi controller
port_name = 'ATOM 0' 

last_midi_msg = None

def midi_listener():
    global last_midi_msg
    try:
        print(f"Opening port: {port_name}...")
        with mido.open_input(port_name) as port:
            for msg in port:
                last_midi_msg = msg
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
        print(f"Error: Could not open port '{port_name}'. Make sure your device is connected and recognized.")
        print("Available ports:")
        print(mido.get_input_names())
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        print("Port closed.")


class TkApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.channel1 = self.channel2 = self.channel3 = self.channel4 = self.channel5 = None
        self.drum_track = self.string_track = self.string_track2 = self.vocal_track = None
        self.s1 = self.s2 = self.s3 = self.s4 = self.s5 = None
        self.button1 = self.button2 = self.button3 = self.button4 = self.button5 = None
        self.drum1_on = self.melody1_on = self.vocal1_on = False
        self.m_thread = Thread(target=self.init_midi_listener, daemon=True)
        self.m_thread.start()
        self.init_tracks()
        self.init_ui()

    def init_tracks(self):
        # add stems to buttons
        pygame.mixer.init(channels=4)

        self.drum_track = pygame.mixer.Sound('doechii_maybe/doechii_maybe_drums.mp3')
        self.string_track = pygame.mixer.Sound('doechii_maybe/doechii_maybe_strings.mp3')
        self.string_track2 = pygame.mixer.Sound('doechii_maybe/doechii_maybe_no_one_mel_pitch.mp3')
        self.vocal_track = pygame.mixer.Sound('doechii_maybe/doechii_maybe_vocals.mp3')

        self.channel1 = pygame.mixer.Channel(0)
        self.channel2 = pygame.mixer.Channel(1)
        self.channel3 = pygame.mixer.Channel(2)
        self.channel4 = pygame.mixer.Channel(3)
        self.channel5 = pygame.mixer.Channel(4)


        self.channel1.play(self.drum_track)
        self.channel2.play(self.string_track)
        self.channel3.play(self.vocal_track)
        # channel4.play(self.drum_track)
        self.channel5.play(self.string_track2)
        self.channel5.set_volume(0)

        # playing bool
        self.ch1_on = True
        self.ch2_on = True
        self.ch3_on = True
        self.ch4_on = True
        self.ch5_on = False

    def init_ui(self):
        self.title('InsideOut')
        self.geometry('500x400')
        self.s1 = tk.Scale(self, from_=1, to=0, resolution=0.01, command=partial(self.update_vol, 'drums'))
        self.s2 = tk.Scale(self, from_=1, to=0, resolution=0.01, command=partial(self.update_vol, 'melody'))
        self.s3 = tk.Scale(self, from_=1, to=0, resolution=0.01, command=partial(self.update_vol, 'vocals'))
        self.s4 = tk.Scale(self, from_=1, to=0, resolution=0.01, command=partial(self.update_vol, 'drums'))
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
        while True:
            global last_midi_msg
            if last_midi_msg is not None:
                if last_midi_msg.type == 'note_on':
                    self.process_midi_input(last_midi_msg.note)
                if last_midi_msg.type == 'note_off':
                    self.drum1_on = False
                    self.melody1_on = False
                    self.vocal1_on = False
                if last_midi_msg.type == 'control_change':
                    self.process_midi_control_change(last_midi_msg)

            time.sleep(0.01)

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

    def update_vol(self, track, value):
        val = 0.01 * value
        match track:
            case 'drums':
                if self.ch1_on:
                    self.channel1.set_volume(float(val))
                    self.s1.set(val)

            case 'melody':
                if self.ch2_on:
                    self.channel2.set_volume(float(val))
                    self.s2.set(val)
                else:
                    self.channel5.set_volume(float(val))
                    self.s2.set(val)
                

            case 'vocals':
                if self.ch3_on:
                    self.channel3.set_volume(float(val))
                    self.s3.set(val)
                
            
            case _:
                print('Invalid volume adjustment')

    def play_drums(self):
        if not self.ch1_on:
            self.channel1.set_volume(self.s1.get())
            self.button1.config(text='Drums (on)')
            self.ch1_on = True
        else:
            self.channel1.set_volume(0)
            self.button1.config(text='Drums (off)')
            self.ch1_on = False

    def play_melody(self, tr):
        match tr:
            # Channel 2
            case 1:
                if not self.ch2_on:
                    self.channel2.set_volume(self.s2.get())
                    self.channel5.set_volume(0)
                    self.ch2_on = True
                    self.ch5_on = False
                else:
                    self.channel2.set_volume(0)
                    self.ch2_on = False
            
            # Channel 5
            case 2:
                if not self.ch5_on:
                    self.channel5.set_volume(self.s2.get())
                    self.channel2.set_volume(0)
                    self.ch2_on = False
                    self.ch5_on = True
                else:
                    self.channel5.set_volume(0)
                    self.ch5_on = False

    def play_vocals(self):
        if not self.ch3_on:
            self.channel3.set_volume(self.s3.get())
            self.button3.config(text='Vocals (on)')
            self.ch3_on = True
        else:
            self.channel3.set_volume(0)
            self.button3.config(text='Vocals (off)')
            self.ch3_on = False

    # def start_drum_hit(e):
    #     print(e)
    #     self.channel1.set_volume(0)
    #     self.channel4.play(self.drum_track)

    # def stop_drum_hit(e):
    #     self.channel1.set_volume(1)
    #     self.channel4.stop()




# button4.bind('<ButtonPress-1>', start_drum_hit)
# button4.bind('<ButtonRelease-1>', stop_drum_hit)
# button4.pack()

if __name__ == '__main__':
    t = Thread(target=midi_listener, daemon=True)
    t.start()
    window = TkApp()
    window.mainloop()
