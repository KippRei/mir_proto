import tkinter as tk
import pygame
from functools import partial

pygame.mixer.init(channels=4)

drum_track = pygame.mixer.Sound('doechii_maybe/doechii_maybe_drums.mp3')
string_track = pygame.mixer.Sound('doechii_maybe/doechii_maybe_strings.mp3')
string_track2 = pygame.mixer.Sound('doechii_maybe/doechii_maybe_no_one_mel_pitch.mp3')
vocal_track = pygame.mixer.Sound('doechii_maybe/doechii_maybe_vocals.mp3')

channel1 = pygame.mixer.Channel(0)
channel2 = pygame.mixer.Channel(1)
channel3 = pygame.mixer.Channel(2)
channel4 = pygame.mixer.Channel(3)
channel5 = pygame.mixer.Channel(4)


channel1.play(drum_track)
channel2.play(string_track)
channel3.play(vocal_track)
# channel4.play(drum_track)
channel5.play(string_track2)
channel5.set_volume(0)

# playing bool
ch1_on = True
ch2_on = True
ch3_on = True
ch4_on = True
ch5_on = False

def update_vol(track, val):
    match track:
        case 'drums':
            if ch1_on:
                channel1.set_volume(float(val))

        case 'melody':
            if ch2_on:
                channel2.set_volume(float(val))
            else:
                channel5.set_volume(float(val))

        case 'vocals':
            if ch3_on:
                channel3.set_volume(float(val))
        
        case _:
            print('Invalid volume adjustment')

a = tk.Tk()
a.title('InsideOut')
a.geometry('500x400')
s1 = tk.Scale(a, from_=1, to=0, resolution=0.01, command=partial(update_vol, 'drums'))
s2 = tk.Scale(a, from_=1, to=0, resolution=0.01, command=partial(update_vol, 'melody'))
s3 = tk.Scale(a, from_=1, to=0, resolution=0.01, command=partial(update_vol, 'vocals'))
s4 = tk.Scale(a, from_=1, to=0, resolution=0.01, command=partial(update_vol, 'drums'))
s1.set(1)
s2.set(1)
s3.set(1)
s4.set(1)




def play_drums():
    global ch1_on
    if not ch1_on:
        channel1.set_volume(s1.get())
        button1.config(text='Drums (on)')
        ch1_on = True
    else:
        channel1.set_volume(0)
        button1.config(text='Drums (off)')
        ch1_on = False

def play_melody(tr):
    global ch2_on
    global ch5_on
    match tr:
        # Channel 2
        case 1:
            if not ch2_on:
                channel2.set_volume(s2.get())
                channel5.set_volume(0)
                ch2_on = True
                ch5_on = False
            else:
                channel2.set_volume(0)
                ch2_on = False
        
        # Channel 5
        case 2:
            if not ch5_on:
                channel5.set_volume(s2.get())
                channel2.set_volume(0)
                ch2_on = False
                ch5_on = True
            else:
                channel5.set_volume(0)
                ch5_on = False

def play_vocals():
    global ch3_on
    if not ch3_on:
        channel3.set_volume(s3.get())
        button3.config(text='Vocals (on)')
        ch3_on = True
    else:
        channel3.set_volume(0)
        button3.config(text='Vocals (off)')
        ch3_on = False

def start_drum_hit(e):
    print(e)
    channel1.set_volume(0)
    channel4.play(drum_track)

def stop_drum_hit(e):
    channel1.set_volume(1)
    channel4.stop()



button1 = tk.Button(a, text='Drums (on)', width=25, pady=20, command=play_drums)
button2 = tk.Button(a, text='Melody 1', width=25, pady=20, command=partial(play_melody, 1))
button3 = tk.Button(a, text='Vocals (on)', width=25, pady=20, command=play_vocals)
button4 = tk.Button(a, text='Drum (hit)', width=25, pady=20)
button5 = tk.Button(a, text='Melody 2', width=25, pady=20, command=partial(play_melody, 2))


button1.grid(row=0, column=0)
button2.grid(row=1, column=0)
button3.grid(row=2, column=0)
button4.grid(row=3, column=0)
button5.grid(row=1, column=2)

s1.grid(row=0, column=1)
s2.grid(row=1, column=1)
s3.grid(row=2, column=1)
s4.grid(row=3, column=1)




# button4.bind('<ButtonPress-1>', start_drum_hit)
# button4.bind('<ButtonRelease-1>', stop_drum_hit)
# button4.pack()
a.mainloop()