"""
The AudioPlayer class is a Python wrapper for the audio engine implemented in Rust
(audio_player.rx). It handles loading and playing preprocessed
audio data and volume adjustment.
"""
import numpy as np
import os
import zmkr_audio_engine

# Wrapper for Rust implementation
class AudioPlayer():
    """
    Manages the audio state. Handles loading songs, playback,
    and track volumes.
    """
    def __init__(self):
        """
        Initializes the Mixer class (audio_player.rs) and calls load_preprocessed_songs to
        load all preprocessed songs from the './preprocessed_audio' directory.
        """
        # To hold volume levels
        self.mixer = zmkr_audio_engine.Mixer()
        self.load_preprocessed_songs()
        self.mixer.start_audio_processing()
        self.is_playing = False

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
                    self.mixer.load_preprocessed_song(os.fsdecode(file_name), song_file_name.split('.')[0], np.load(f"{path}/{song_file_name}").astype(np.float64))

    # Calls mixer play function to start playing audio.
    def hit_play(self):
        self.is_playing = True
        self.mixer.play()
    # Calls mixer stop function to stop playing audio.
    def hit_stop(self):
        self.is_playing = False
        self.mixer.stop()

    # Adjust volume of drum, bass, melody, or vocal tracks
    def adj_track_vol(self, track, adjustment):
        """
        Adjusts the volume of a specific track.
 
        Parameters
        ----------
        track : str
            The name of the stem/track type ('drum', 'bass', 'melody', or 'vocal').
        adjustment : float
            The amount to change the volume.
        """
        self.mixer.adj_track_vol(track, adjustment)

    # Turn on/off channel and turn off all other channels in its column
    def play_channel(self, channel):
        """
        Toggles the playback status (on/off) of a specific channel pad and
        turns off other channels in the same column (stem type)
        based on the Mixer's logic.
        
        Parameters
        ----------
        channel : int
            The channel number (0-15).
            Pad channels = 
            12 13 14 15
            08 09 10 11
            04 05 06 07
            00 01 02 03
        """
        return self.mixer.channel_on_off(channel)

    # Get track volumes (GUI)
    def get_track_vol(self, name):
        """
        Gets the current volume level of the specified track type.
        
        Parameters
        ----------
        name : str
            The name of the track type ('drum', 'bass', 'melody', or 'vocal').
        
        Returns
        -------
        float
            The current volume (0.0 to 1.0)
        """
        return self.mixer.get_track_vol(name)
            
    def get_channel_list_on_off(self):
        """
        Retrieves the current on/off state of all 16 channels.
        
        Returns
        -------
        list[bool]
            The on/off states of each channel (list).
        """
        return self.mixer.get_channel_list_on_off()
    
    # Gets song list for displaying songs (GUI)
    def get_song_list(self):
        """
        Gets the list of available song titles that have been preprocessed
        and loaded into the mixer.
        
        Returns
        -------
        list[str]
            The list of song titles.
        """
        return self.mixer.get_song_list()
    
    # Loads specified channel with specified track from a song (drum, bass, melody, or vocal)
    def load_track(self, title, channel) -> bool:
        """
        Loads the specified track into a channel.
        
        Parameters
        ----------
        title : str
            The title of the song that is loaded into the channel.
        channel : int
            The channel number in which to load the stem.
        
        Returns
        -------
        bool
            True if the track was successfully loaded and False is not loaded (the track should never fail to load).
        """
        return self.mixer.load_track(title, channel)