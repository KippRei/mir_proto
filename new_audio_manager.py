import sounddevice as sd
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

        self.drum_tr = np.load("./preprocessed_audio/0.npy")
        self.melody_tr = np.load("./preprocessed_audio/2.npy")
        self.vocal_tr = np.load("./preprocessed_audio/3.npy")

        self.mixer = mkr_audio.Mixer()

        self.channel1 = self.mixer.channel(0)
        self.channel2 = self.mixer.channel(1)
        self.channel3 = self.mixer.channel(2)
        self.channel4 = self.mixer.channel(3)
        self.channel5 = self.mixer.channel(4)
        self.channel6 = self.mixer.channel(5)
        self.channel7 = self.mixer.channel(6)
        self.channel8 = self.mixer.channel(7)
        self.channel9 = self.mixer.channel(8)
        self.channel10 = self.mixer.channel(9)
        self.channel11 = self.mixer.channel(10)
        self.channel12 = self.mixer.channel(11)
        self.channel13 = self.mixer.channel(12)
        self.channel14 = self.mixer.channel(13)
        self.channel15 = self.mixer.channel(14)
        self.channel16 = self.mixer.channel(15)

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

        # for v in self.channel_map.values():
        #     v.set_volume(0)
        self.channel1.load(self.drum_tr)
        self.channel3.load(self.melody_tr)
        self.channel4.load(self.vocal_tr)

        # channel4.play(self.drum_track)

    def hit_play(self):
        self.mixer.play()
    
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
        if self.mixer.channel_on[channel] is False:
           self.mixer.channel_on[channel] = True
        else: 
            col_mod = channel % 4
            for k, v in self.channel_map.items():
                if k % 4 == col_mod:
                    v.set_volume(0) 

            match channel:
                case 1:
                    self.mixer.channel_on[1] = True
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
    

# if __name__ == "__main__":
#     a = AudioPlayer()