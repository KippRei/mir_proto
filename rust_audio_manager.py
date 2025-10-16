import numpy as np
import os
import zmkr_audio_engine

# Wrapper for Rust implementation
class AudioPlayer():
    def __init__(self):
        # To hold volume levels
        self.mixer = zmkr_audio_engine.Mixer()
        self.load_preprocessed_songs()
        self.mixer.add_to_playback("apollo")

    # Loads all the preprocessed songs into mixer
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
                    self.mixer.load_preprocessed_song(os.fsdecode(file_name), song_file_name.split('.')[0], np.load(f"{path}/{song_file_name}").astype(np.float32))

        # self.mixer.print_song_map()

    def hit_play(self):
        # TODO
        self.mixer.play()
    
    def hit_stop(self):
        # TODO
        self.mixer.stop()

    # Adjust volume of drum, bass, melody, or vocal tracks
    def adj_track_vol(self, track, adjustment):
        self.mixer.adj_track_vol(track, adjustment)

    # Turn on/off channel and turn off all other channels in its column
    def play_channel(self, channel):
        self.mixer.channel_on_off(channel)

    # Get track volumes (GUI)
    def get_track_vol(self, name):
        return self.mixer.get_track_vol(name)
            
    def get_channel_list_on_off(self):
        # TODO
        return self.mixer.get_channel_list_on_off()
    
    # Gets song list for displaying songs (GUI)
    def get_song_list(self):
        return self.mixer.get_song_list()
    
    # Loads specified channel with specified track from a song (drum, bass, melody, or vocal)
    def load_track(self, title, channel) -> bool:
        return self.mixer.load_track(title, channel)