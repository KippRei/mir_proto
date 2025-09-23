import numpy as np
import mkr_audio

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

        self.drum_tr_1 = np.load("./preprocessed_audio/doechii_maybe_drum.npy")
        self.melody_tr_1 = np.load("./preprocessed_audio/doechii_maybe_strings.npy")
        self.melody_tr_2 = np.load("./preprocessed_audio/doechii_maybe_strings.npy")
        self.vocal_tr_1 = np.load("./preprocessed_audio/doechii_maybe_vocal.npy")
        self.vocal_tr_2 = np.load("./preprocessed_audio/doechii_sos_vocal.npy")
        self.melody_tr_3 = np.load("./preprocessed_audio/doechii_sos_melody.npy")
        self.drum_tr_2 = np.load("./preprocessed_audio/doechii_sos_drum.npy")
        self.bass_tr_1 = np.load("./preprocessed_audio/doechii_sos_bass.npy")


        self.mixer = mkr_audio.Mixer()
        self.channel0 = self.mixer.channel(0)
        self.channel1 = self.mixer.channel(1)
        self.channel2 = self.mixer.channel(2)
        self.channel3 = self.mixer.channel(3)
        self.channel4 = self.mixer.channel(4)
        self.channel5 = self.mixer.channel(5)
        self.channel6 = self.mixer.channel(6)
        self.channel7 = self.mixer.channel(7)
        self.channel8 = self.mixer.channel(8)
        self.channel9 = self.mixer.channel(9)
        self.channel10 = self.mixer.channel(10)
        self.channel11 = self.mixer.channel(11)
        self.channel12 = self.mixer.channel(12)
        self.channel13 = self.mixer.channel(13)
        self.channel14 = self.mixer.channel(14)
        self.channel15 = self.mixer.channel(15)

        self.channel0.load(self.drum_tr_1)
        self.channel1.load(self.bass_tr_1)
        self.channel3.load(self.vocal_tr_1)
        self.channel4.load(self.drum_tr_2)
        self.channel6.load(self.melody_tr_2)
        self.channel7.load(self.vocal_tr_2)
        self.channel10.load(self.melody_tr_3)

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
        return self.mixer.channel_map
    

# if __name__ == "__main__":
#     a = AudioPlayer()