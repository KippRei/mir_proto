import time
import sounddevice as sd
import librosa
import numpy as np

# File paths for all the stems
fp = 'doechii_maybe/doechii_maybe_drums.mp3'
fp2 = 'doechii_maybe/doechii_maybe_strings.mp3'
fp3 = 'doechii_maybe/doechii_maybe_no_one_mel_pitch.mp3'
fp4 = 'doechii_maybe/doechii_maybe_vocals.mp3'

# Load all audio files at a high sample rate and preserve stereo channels
# The .T transposes the array from (channels, samples) to (samples, channels)
# This is required by sounddevice
y, sr = librosa.load(fp, sr=44100, mono=False)
y1, sr = librosa.load(fp2, sr=44100, mono=False)
y2, sr = librosa.load(fp3, sr=44100, mono=False)
y3, sr = librosa.load(fp4, sr=44100, mono=False)

# Get the length of the longest array. The length is the number of samples,
# which is the second dimension of the array.
max_len = max(y.shape[1], y1.shape[1], y2.shape[1], y3.shape[1])

# Pad each array with zeros to match the maximum length
# The pad_width is a tuple of tuples: ((rows before, rows after), (cols before, cols after))
padded_y = np.pad(y, ((0, 0), (0, max_len - y.shape[1])))
padded_y1 = np.pad(y1, ((0, 0), (0, max_len - y1.shape[1])))
padded_y2 = np.pad(y2, ((0, 0), (0, max_len - y2.shape[1])))
padded_y3 = np.pad(y3, ((0, 0), (0, max_len - y3.shape[1])))

# Now that all arrays have the same length, you can add them together
Y = padded_y + padded_y1 + padded_y3

# Normalize the combined signal to prevent clipping
normalized_Y = Y / len([y, y1, y3])

# Transpose the array to the (samples, channels) format
# required by sounddevice for stereo playback
final_audio = np.transpose(Y)

# Play the combined audio and wait
sd.play(final_audio, sr)
sd.wait()

class AudioPlayer():
    def __init__(self):
        self.drum_vol = self.bass_vol = self.melody_vol = self.vocal_vol = 1
                # add stems to buttons

        self.drum_track = pygame.mixer.Sound('doechii_maybe/doechii_maybe_drums.mp3')
        self.string_track = pygame.mixer.Sound('doechii_maybe/doechii_maybe_strings.mp3')
        self.string_track2 = pygame.mixer.Sound('doechii_maybe/doechii_maybe_no_one_mel_pitch.mp3')
        self.vocal_track = pygame.mixer.Sound('doechii_maybe/doechii_maybe_vocals.mp3')

        self.channel1 = pygame.mixer.Channel(0)
        self.channel2 = pygame.mixer.Channel(1)
        self.channel3 = pygame.mixer.Channel(2)
        self.channel4 = pygame.mixer.Channel(3)
        self.channel5 = pygame.mixer.Channel(4)
        self.channel6 = pygame.mixer.Channel(5)
        self.channel7 = pygame.mixer.Channel(6)
        self.channel8 = pygame.mixer.Channel(7)
        self.channel9 = pygame.mixer.Channel(8)
        self.channel10 = pygame.mixer.Channel(9)
        self.channel11 = pygame.mixer.Channel(10)
        self.channel12 = pygame.mixer.Channel(11)
        self.channel13 = pygame.mixer.Channel(12)
        self.channel14 = pygame.mixer.Channel(13)
        self.channel15 = pygame.mixer.Channel(14)
        self.channel16 = pygame.mixer.Channel(15)

        self.channel_map = {
            1: self.channel1,
            2: self.channel2,
            3: self.channel3,
            4: self.channel4,
            5: self.channel5,
            6: self.channel6,
            7: self.channel7,
            8: self.channel8,
            9: self.channel9,
            10: self.channel10,
            11: self.channel11,
            12: self.channel12,
            13: self.channel13,
            14: self.channel14,
            15: self.channel15,
            16: self.channel16,
        }
        for v in self.channel_map.values():
            v.set_volume(0)
        self.channel1.queue(self.drum_track)
        self.channel3.queue(self.string_track)
        self.channel4.queue(self.vocal_track)
        # channel4.play(self.drum_track)
        self.channel7.queue(self.string_track2)
        self.hit_stop()

    def hit_play(self):
        for v in self.channel_map.values():
            v.stop()
        # Uncomment this to autoplay when program starts
        self.channel1.queue(self.drum_track)
        self.channel3.queue(self.string_track)
        self.channel4.queue(self.vocal_track)
        # channel4.play(self.drum_track)
        self.channel7.queue(self.string_track2)
    
    def hit_stop(self):
        for v in self.channel_map.values():
            v.stop()

    def adj_track_vol(self, track, adjustment):
        # TODO: made 1 the lowest volume so I can set volume 0 to indicate button on or off
        # TODO (cont): in the play_channel method
        # TODO (cont): hacky way of doing this, fix it after done using it for testing
        match track:
            case 'drum':
                self.drum_vol += adjustment
                if self.drum_vol > 1:
                    self.drum_vol = 1
                elif self.drum_vol < 0.01:
                    self.drum_vol = 0.01
            case 'bass':
                self.bass_vol += adjustment
                if self.bass_vol > 1:
                    self.bass_vol = 1
                elif self.bass_vol < 0.01:
                    self.bass_vol = 0.01
            case 'melody':
                self.melody_vol += adjustment
                if self.melody_vol > 1:
                    self.melody_vol = 1
                elif self.melody_vol < 0.01:
                    self.melody_vol = 0.01
            case 'vocal':
                self.vocal_vol += adjustment
                if self.vocal_vol > 1:
                    self.vocal_vol = 1
                elif self.vocal_vol < 0.01:
                    self.vocal_vol = 0.01
        self.update_vol(track)

    # TODO: can be more effecient (not loop over every item in channel_map)
    def update_vol(self, channel_type):
        channel_mod = -1
        vol = None
        match channel_type:
            case 'drum':
                channel_mod = 1
                vol = self.drum_vol
            case 'bass':
                channel_mod = 2
                vol = self.bass_vol
            case 'melody':
                channel_mod = 3
                vol = self.melody_vol
            case 'vocal':
                channel_mod = 0
                vol = self.vocal_vol

        for k, v in self.channel_map.items():
            if k % 4 == channel_mod and v.get_volume() != 0:
                v.set_volume(vol)        

    def play_channel(self, channel):
        # Mute all channels in column then unmute the one that was selected
        # TODO: Do this a better way
        # TODO: to see if track/pad/light is on or off we check to see if volume is 0
        # TODO (cont): hacky way of doing this, fix after testing
        if self.channel_map[channel].get_volume() != 0:
           self.channel_map[channel].set_volume(0)
        else: 
            col_mod = channel % 4
            for k, v in self.channel_map.items():
                if k % 4 == col_mod:
                    v.set_volume(0) 

            match channel:
                case 1:
                    self.channel1.set_volume(self.drum_vol)
                case 2:
                    self.channel2.set_volume(self.bass_vol)
                case 3:
                    self.channel3.set_volume(self.melody_vol)
                case 4:
                    self.channel4.set_volume(self.vocal_vol)
                case 5:
                    self.channel5.set_volume(self.drum_vol)
                case 6:
                    self.channel6.set_volume(self.bass_vol)
                case 7:
                    self.channel7.set_volume(self.melody_vol)
                case 8:
                    self.channel8.set_volume(self.vocal_vol)
                case 9:
                    self.channel9.set_volume(self.drum_vol)
                case 10:
                    self.channel10.set_volume(self.bass_vol)
                case 11:
                    self.channel11.set_volume(self.melody_vol)
                case 12:
                    self.channel12.set_volume(self.vocal_vol)
                case 13:
                    self.channel13.set_volume(self.drum_vol)
                case 14:
                    self.channel14.set_volume(self.bass_vol)
                case 15:
                    self.channel15.set_volume(self.melody_vol)
                case 16:
                    self.channel16.set_volume(self.vocal_vol)


    def get_track_vol(self, name):
        match name:
            case 'drum':
                return self.drum_vol
            case 'bass':
                return self.bass_vol
            case 'melody':
                return self.melody_vol
            case 'vocal':
                return self.vocal_vol
            
    def get_channel_map(self):
        return self.channel_map