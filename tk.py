import tkinter as tk
import pygame

pygame.mixer.init(channels=4)

drum_track = pygame.mixer.Sound('doechii_maybe/doechii_maybe_drums.mp3')
string_track = pygame.mixer.Sound('doechii_maybe/doechii_maybe_strings.mp3')
vocal_track = pygame.mixer.Sound('doechii_maybe/doechii_maybe_vocals.mp3')

channel1 = pygame.mixer.Channel(0)
channel2 = pygame.mixer.Channel(1)
channel3 = pygame.mixer.Channel(2)
channel4 = pygame.mixer.Channel(3)

channel1.play(drum_track)
channel2.play(string_track)
channel3.play(vocal_track)

def play_drums():
    if channel1.get_volume() == 0:
        channel1.set_volume(1)
        button1.config(text='Drums (on)')
    else:
        channel1.set_volume(0)
        button1.config(text='Drums (off)')

def play_strings():
    if channel2.get_volume() == 0:
        channel2.set_volume(1)
        button2.config(text='Strings (on)')
    else:
        channel2.set_volume(0)
        button2.config(text='Strings (off)')

def play_vocals():
    if channel3.get_volume() == 0:
        channel3.set_volume(1)
        button3.config(text='Vocals (on)')
    else:
        channel3.set_volume(0)
        button3.config(text='Vocals (off)')

a = tk.Tk()
a.title('InsideOut')
a.geometry('400x300')
button1 = tk.Button(a, text='Drums (on)', width=25, pady=20, command=play_drums)
button1.pack()
button2 = tk.Button(a, text='Strings (on)', width=25, pady=20, command=play_strings)
button2.pack()
button3 = tk.Button(a, text='Vocals (on)', width=25, pady=20, command=play_vocals)
button3.pack()
a.mainloop()