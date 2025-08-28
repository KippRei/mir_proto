import sys
import threading
import time
import mido
import pygame
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
from PyQt6.QtCore import QTimer

# --- Global Variables for MIDI Communication ---
# The ATOM's default note numbers for the main 16 pads
MIDI_PAD_DRUMS = 36  # Pad 1
MIDI_PAD_STRINGS = 37  # Pad 2
MIDI_PAD_VOCALS = 38  # Pad 3
MIDI_PAD_DRUM_HIT = 39 # Pad 4

# Pad color control numbers (CC numbers)
CC_PAD_DRUMS = 16
CC_PAD_STRINGS = 17
CC_PAD_VOCALS = 18

# Color values
COLOR_OFF = 0
COLOR_RED = 1
COLOR_GREEN = 3

# Global variables for communication between threads
last_message = None
out_port = None

def midi_listener():
    """Listens for MIDI messages in a separate thread."""
    global last_message
    input_port_name = 'ATOM 0'  # Use the correct port name for your device
    
    try:
        with mido.open_input(input_port_name) as port:
            print(f"Listening for MIDI messages on '{input_port_name}' in a separate thread.")
            for msg in port:
                last_message = msg
                print(f"Thread received: {msg}")
    except mido.PortNotOpenError:
        print(f"Error: Could not open port '{input_port_name}'. Check device connection.")
    except Exception as e:
        print(f"An unexpected error occurred in the MIDI listener: {e}")

# --- Pygame and Audio Setup ---
pygame.mixer.init(channels=4)

try:
    drum_track = pygame.mixer.Sound('doechii_maybe/doechii_maybe_drums.mp3')
    string_track = pygame.mixer.Sound('doechii_maybe/doechii_maybe_strings.mp3')
    vocal_track = pygame.mixer.Sound('doechii_maybe/doechii_maybe_vocals.mp3')
    drum_hit = pygame.mixer.Sound('doechii_maybe/doechii_maybe_drums.mp3')
except pygame.error as e:
    print(f"Error loading sound files: {e}")
    sys.exit()

channel_drums = pygame.mixer.Channel(0)
channel_strings = pygame.mixer.Channel(1)
channel_vocals = pygame.mixer.Channel(2)
channel_drum_hit = pygame.mixer.Channel(3)

channel_drums.play(drum_track, loops=-1)
channel_strings.play(string_track, loops=-1)
channel_vocals.play(vocal_track, loops=-1)

# --- PyQt GUI Application ---
class InsideOutApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('InsideOut')
        self.setGeometry(100, 100, 400, 300)
        self.init_ui()
        self.init_midi_timer()

    def init_ui(self):
        layout = QVBoxLayout()
        self.button1 = QPushButton('Drums (on)')
        self.button1.clicked.connect(self.toggle_drums)
        self.button1.setFixedSize(200, 50)
        layout.addWidget(self.button1)
        
        self.button2 = QPushButton('Strings (on)')
        self.button2.clicked.connect(self.toggle_strings)
        self.button2.setFixedSize(200, 50)
        layout.addWidget(self.button2)
        
        self.button3 = QPushButton('Vocals (on)')
        self.button3.clicked.connect(self.toggle_vocals)
        self.button3.setFixedSize(200, 50)
        layout.addWidget(self.button3)
        self.setLayout(layout)

    def init_midi_timer(self):
        self.midi_timer = QTimer(self)
        self.midi_timer.setInterval(20)
        self.midi_timer.timeout.connect(self.process_midi_messages)
        self.midi_timer.start()

    def process_midi_messages(self):
        global last_message, out_port
        if last_message is not None:
            msg = last_message
            last_message = None

            if msg.type == 'note_on':
                if msg.note == MIDI_PAD_DRUMS:
                    self.toggle_drums()
                    if channel_drums.get_volume() > 0:
                        self.send_midi_color(CC_PAD_DRUMS, COLOR_GREEN)
                    else:
                        self.send_midi_color(CC_PAD_DRUMS, COLOR_OFF)
                elif msg.note == MIDI_PAD_STRINGS:
                    self.toggle_strings()
                    if channel_strings.get_volume() > 0:
                        self.send_midi_color(CC_PAD_STRINGS, COLOR_GREEN)
                    else:
                        self.send_midi_color(CC_PAD_STRINGS, COLOR_OFF)
                elif msg.note == MIDI_PAD_VOCALS:
                    self.toggle_vocals()
                    if channel_vocals.get_volume() > 0:
                        self.send_midi_color(CC_PAD_VOCALS, COLOR_GREEN)
                    else:
                        self.send_midi_color(CC_PAD_VOCALS, COLOR_OFF)
                elif msg.note == MIDI_PAD_DRUM_HIT:
                    channel_drum_hit.play(drum_hit)
            elif msg.type == 'note_off':
                if msg.note == MIDI_PAD_DRUM_HIT:
                    channel_drum_hit.stop()
                    
    def send_midi_color(self, control, value):
        global out_port
        if out_port:
            msg = mido.Message('control_change', channel=0, control=control, value=value)
            try:
                out_port.send(msg)
                print(f"Sent MIDI CC message: {msg}")
            except Exception as e:
                print(f"Failed to send MIDI message: {e}")

    def toggle_drums(self):
        if channel_drums.get_volume() > 0:
            channel_drums.set_volume(0)
            self.button1.setText('Drums (off)')
        else:
            channel_drums.set_volume(1)
            self.button1.setText('Drums (on)')

    def toggle_strings(self):
        if channel_strings.get_volume() > 0:
            channel_strings.set_volume(0)
            self.button2.setText('Strings (off)')
        else:
            channel_strings.set_volume(1)
            self.button2.setText('Strings (on)')

    def toggle_vocals(self):
        if channel_vocals.get_volume() > 0:
            channel_vocals.set_volume(0)
            self.button3.setText('Vocals (off)')
        else:
            channel_vocals.set_volume(1)
            self.button3.setText('Vocals (on)')

# --- Main Application Entry Point ---
if __name__ == '__main__':
    input_port_name = 'ATOM 0'
    output_port_name = 'ATOM 1'
    print(f"Available MIDI input ports: {mido.get_input_names()}")
    print(f"Available MIDI output ports: {mido.get_output_names()}")

    try:
        out_port = mido.open_output(output_port_name)
    except Exception as e:
        print(f"Error opening MIDI output port: {e}")
        # sys.exit(1)

    midi_thread = threading.Thread(target=midi_listener, daemon=True)
    midi_thread.start()
    
    # Initialize the button colors
    try:
        out_port.send(mido.Message('control_change', channel=0, control=CC_PAD_DRUMS, value=COLOR_GREEN))
        out_port.send(mido.Message('control_change', channel=0, control=CC_PAD_STRINGS, value=COLOR_GREEN))
        out_port.send(mido.Message('control_change', channel=0, control=CC_PAD_VOCALS, value=COLOR_GREEN))
    except Exception as e:
        print(f"Error initializing pad colors: {e}")

    app = QApplication(sys.argv)
    window = InsideOutApp()
    window.show()

    sys.exit(app.exec())