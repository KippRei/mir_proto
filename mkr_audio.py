import numpy as np
import sounddevice as sd

class Mixer():
    curr_frame = 0
    def __init__(self, channels: int= 16):
        self.loop_length = 60000 # need to figure out how long 32 bars is
        self.channel_map = {}
        self.channel_on = {}
        self.currently_playing = np.zeros(shape=(4351282, 2), dtype=float)
        for i in range(channels):
            self.channel_map[i] = _Channel(i, self)
            self.channel_on[i] = False

    def channel(self, channel_num):
        return self.channel_map[channel_num]
    
    # for testing all audio playback
    def update_playing(self):
        self.currently_playing = np.zeros(shape=(4351282, 2), dtype=float)
        for k, v in self.channel_on.items():
            if v is True:
                self.currently_playing += self.channel_map[k].get_data()
    
    def play(self):
        with sd.OutputStream(samplerate=44100, callback=self.audio_callback, channels=2) as stream:
            sd.sleep(self.loop_length)

    def audio_callback(self, outdata, frames, time, status):
        outdata[:frames] = self.currently_playing[self.curr_frame: self.curr_frame + frames]
        self.curr_frame += frames
        
class _Channel():
    def __init__(self, channel_num, mixer):
        self.channel_num = channel_num
        self.mixer = mixer
        self.volume = 1
        self.data = np.zeros(shape=(4351282, 2), dtype=float)

    def load(self, data=np.ndarray):
        self.data = data

    def on(self):
        self.mixer.channel_on[self.channel_num] = True
        self.mixer.update_playing()

    def off(self):
        self.mixer.channel_on[self.channel_num] = False
        self.mixer.update_playing()

    # TODO: implement volume controls
    def get_volume(self):
        return self.volume
    
    def set_volume(self, vol):
        self.volume = vol

    # For testing
    def get_data(self):
        return self.data