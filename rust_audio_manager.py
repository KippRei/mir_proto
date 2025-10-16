import numpy as np
import mkr_audio
import os
import zmkr_audio_engine

# Y = padded_y + padded_y1 + padded_y3

# # Normalize the combined signal to prevent clipping
# # normalized_Y = Y / len([y, y1, y3])

# # Transpose the array to the (samples, channels) format
# # required by sounddevice for stereo playback
# final_audio = np.transpose(Y)

# # Play the combined audio and wait
# sd.play(final_audio, 44100)
# sd.wait()

class AudioPlayer():
    def __init__(self):
        # To hold volume levels
        self.drum_vol = self.bass_vol = self.melody_vol = self.vocal_vol = 1

        self.mixer = zmkr_audio_engine.Mixer()
        self.load_preprocessed_songs()


        # self.channel0 = self.mixer.channel(0)
        # self.channel1 = self.mixer.channel(1)
        # self.channel2 = self.mixer.channel(2)
        # self.channel3 = self.mixer.channel(3)
        # self.channel4 = self.mixer.channel(4)
        # self.channel5 = self.mixer.channel(5)
        # self.channel6 = self.mixer.channel(6)
        # self.channel7 = self.mixer.channel(7)
        # self.channel8 = self.mixer.channel(8)
        # self.channel9 = self.mixer.channel(9)
        # self.channel10 = self.mixer.channel(10)
        # self.channel11 = self.mixer.channel(11)
        # self.channel12 = self.mixer.channel(12)
        # self.channel13 = self.mixer.channel(13)
        # self.channel14 = self.mixer.channel(14)
        # self.channel15 = self.mixer.channel(15)

    def load_preprocessed_songs(self):
        # Load all preprocessed songs into songs map
        song_dir = './preprocessed_audio'
        song_dir_encoded = os.fsencode(song_dir)
        for file_name in os.listdir(song_dir_encoded):
            path = f"{song_dir}/{os.fsdecode(file_name)}"
            curr_dir = os.fsencode(path)
            if os.path.isdir(curr_dir):
                for file in os.listdir(curr_dir):
                    song_file_name = os.fsdecode(file)
                    self.mixer.load_preprocessed_song(os.fsdecode(file_name), song_file_name.split('.')[0], np.load(f"{path}/{song_file_name}").astype(np.float64))

        self.mixer.print_song_map()
        # self.mixer.play("The BoykinZ - Step Right Up (Official Music Video)")

    def hit_play(self):
        self.mixer.play()
    
    def hit_stop(self):
        self.mixer.stop()

    def adj_track_vol(self, track, adjustment):
        match track:
            case 'drum':
                self.drum_vol += adjustment
                if self.drum_vol > 1:
                    self.drum_vol = 1
                elif self.drum_vol < 0:
                    self.drum_vol = 0
                for i in range(0,16,4):
                    self.mixer.channel_map[i]['volume'] = self.drum_vol
            case 'bass':
                self.bass_vol += adjustment
                if self.bass_vol > 1:
                    self.bass_vol = 1
                elif self.bass_vol < 0:
                    self.bass_vol = 0
                for i in range(1,16,4):
                    self.mixer.channel_map[i]['volume'] = self.bass_vol
            case 'melody':
                self.melody_vol += adjustment
                if self.melody_vol > 1:
                    self.melody_vol = 1
                elif self.melody_vol < 0:
                    self.melody_vol = 0
                for i in range(2,16,4):
                    self.mixer.channel_map[i]['volume'] = self.melody_vol
            case 'vocal':
                self.vocal_vol += adjustment
                if self.vocal_vol > 1:
                    self.vocal_vol = 1
                elif self.vocal_vol < 0:
                    self.vocal_vol = 0
                for i in range(3,16,4):
                    self.mixer.channel_map[i]['volume'] = self.vocal_vol

    # Turn on/off channel and turn off all other channels in its column
    def play_channel(self, channel):
        self.mixer.channel_map[channel]['is_playing'] = not self.mixer.channel_map[channel]['is_playing']
        col_mod = channel % 4
        for k in self.mixer.channel_map.keys():
            if k % 4 == col_mod and k != channel:
                self.mixer.channel_map[k]['is_playing'] = False

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
        return {}
    
    def get_songs_map(self):
        return {}
    
    def load_track(self, title, channel) -> bool:
        if title not in self.songs_map.keys():
            return False
        song = self.songs_map[title]
        match channel:
            case 0:
                if 'drum' in song:
                    self.channel0.load(song['drum'])
                    return True
            case 1:
                if 'bass' in song:
                    self.channel1.load(song['bass'])
                    return True
            case 2:
                if 'melody' in song:
                    self.channel2.load(song['melody'])
                    return True
            case 3:
                if 'vocal' in song:
                    self.channel3.load(song['vocal'])
                    return True
                    
            case 4:
                if 'drum' in song:
                    self.channel4.load(song['drum'])
                    return True
            case 5:
                if 'bass' in song:
                    self.channel5.load(song['bass'])
                    return True
            case 6:
                if 'melody' in song:
                    self.channel6.load(song['melody'])
                    return True
            case 7:
                if 'vocal' in song:
                    self.channel7.load(song['vocal'])
                    return True
                    
            case 8:
                if 'drum' in song:
                    self.channel8.load(song['drum'])
                    return True
            case 9:
                if 'bass' in song:
                    self.channel9.load(song['bass'])
                    return True
            case 10:
                if 'melody' in song:
                    self.channel10.load(song['melody'])
                    return True
            case 11:
                if 'vocal' in song:
                    self.channel11.load(song['vocal'])
                    return True
                    
            case 12:
                if 'drum' in song:
                    self.channel12.load(song['drum'])
                    return True
            case 13:
                if 'bass' in song:
                    self.channel13.load(song['bass'])
                    return True
            case 14:
                if 'melody' in song:
                    self.channel14.load(song['melody'])
                    return True
            case 15:
                if 'vocal' in song:
                    self.channel15.load(song['vocal'])
                    return True
            
            case _:
                return False