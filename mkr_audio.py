# This class acts as a 

import numpy as np
import sounddevice as sd
import threading

class Mixer():
    curr_frame = 0
    def __init__(self, channels: int= 16):
        self.is_playing = False # flag for whether audio is playing or not
        self.mix_lock = threading.Lock()
        self.channel_map = {}
        self.stop_event = threading.Event() # this signal is for stopping audio_thread
        self.currently_playing = np.zeros(shape=(4351282 * 2, 2), dtype=float)
        for i in range(0, channels):
            self.channel_map[i] = {
                'channel': _Channel(i, self),
                'volume': 1.0,
                'is_playing': False
            }

    def channel(self, channel_num):
        return self.channel_map[channel_num]['channel']
    
    def play(self):
        if not self.is_playing:
            self.is_playing = True
            self.stop_event.clear()
            self.audio_thread = threading.Thread(target=self.__start_playing, daemon=True)
            self.audio_thread.start()
    
    def stop(self):
        if self.is_playing:
                self.is_playing = False
                self.__stop_playing()

    def __start_playing(self):
        with sd.OutputStream(samplerate=44100, callback=self.audio_callback, channels=2) as stream:
            # The audio will loop so we want it to continue playing until user presses stop
            while True and self.is_playing:
                pass

    def __stop_playing(self):
        self.stop_event.set() # set flag to true (to stop playback)
        self.curr_frame = 0
        # self.audio_thread.join() # blocks calling thread (essentially main) until playback stopped

    def audio_callback(self, outdata, frames, time, status):
        # if stop button was pressed (using raise sd.CallbackStop as required by sd library)
        if self.stop_event.is_set():
            raise sd.CallbackStop
        
        with self.mix_lock:
            # TODO: prevents buffer overflow but need to loop intelligently (on beat) rather than like this
            if self.curr_frame + frames > self.currently_playing.shape[0]:
                self.curr_frame = 0

            temp_buffer = np.zeros(shape=(frames, 2), dtype=float)
            for v in self.channel_map.values():
                if v['is_playing']:
                    temp_buffer[:frames] += (v['channel'].get_data()[self.curr_frame:self.curr_frame + frames] * v['volume'])
        
            np.clip(temp_buffer, -1.0, 1.0, out=outdata) # clamps between -1.0 and 1.0 to prevent audio clipping

            outdata[:] = temp_buffer
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
        self.mixer.channel_map[self.channel_num]['is_playing'] = True
        mix_update_thread = threading.Thread(target=self.mixer.update_playing, daemon=True)
        mix_update_thread.start()

    def off(self):
        self.mixer.channel_map[self.channel_num]['is_playing'] = False
        mix_update_thread = threading.Thread(target=self.mixer.update_playing, daemon=True)
        mix_update_thread.start()

    # TODO: implement volume controls
    def get_volume(self):
        return self.volume
    
    def set_volume(self, vol):
        self.volume = vol

    # For testing
    def get_data(self):
        return self.data