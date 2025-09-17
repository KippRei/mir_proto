import time
import sounddevice as sd
import librosa
import numpy as np

# --- 1. State Management Class ---
class AudioMixer:
    def __init__(self, tracks, sample_rate):
        self.tracks = tracks
        self.active_tracks = {name: True for name in tracks}
        self.current_frame = 0
        self.sample_rate = sample_rate

    def toggle_track(self, name):
        self.active_tracks[name] = not self.active_tracks.get(name, False)
        print(f"Toggled {name} to {self.active_tracks[name]}")

    def audio_callback(self, outdata, frames, time, status):
        outdata.fill(0)
        
        active_count = 0
        for name, is_active in self.active_tracks.items():
            if is_active:
                track = self.tracks[name]
                active_count += 1
                
                end_frame = self.current_frame + frames
                
                if end_frame > track.shape[0]:
                    remaining_frames = track.shape[0] - self.current_frame
                    outdata[:remaining_frames] += track[self.current_frame:]
                    
                    loop_frames = frames - remaining_frames
                    outdata[remaining_frames:] += track[:loop_frames]
                else:
                    outdata[:] += track[self.current_frame:end_frame]
        
        if active_count > 0:
            outdata /= active_count

        self.current_frame += frames
        longest_track_len = max(t.shape[0] for t in self.tracks.values())
        self.current_frame %= longest_track_len


# --- 2. Main Execution ---
if __name__ == "__main__":
    fp = 'doechii_maybe/doechii_maybe_drums.mp3'
    fp2 = 'doechii_maybe/doechii_maybe_strings.mp3'
    fp3 = 'doechii_maybe/doechii_maybe_no_one_mel_pitch.mp3'
    fp4 = 'doechii_maybe/doechii_maybe_vocals.mp3'

    try:
        # Load audio data and sample rate separately
        y_drums, sr = librosa.load(fp, sr=44100, mono=False)
        y_strings, _ = librosa.load(fp2, sr=44100, mono=False)
        y_mel_pitch, _ = librosa.load(fp3, sr=44100, mono=False)
        y_vocals, _ = librosa.load(fp4, sr=44100, mono=False)
        
        # Transpose each NumPy array individually
        y_drums = y_drums.T
        y_strings = y_strings.T
        y_mel_pitch = y_mel_pitch.T
        y_vocals = y_vocals.T

    except FileNotFoundError:
        print("One or more audio files not found. Please check file paths.")
        exit()

    max_len = max(
        y_drums.shape[0], 
        y_strings.shape[0], 
        y_mel_pitch.shape[0], 
        y_vocals.shape[0]
    )

    padded_tracks = {
        'drums': np.pad(y_drums, ((0, max_len - y_drums.shape[0]), (0, 0))),
        'strings': np.pad(y_strings, ((0, max_len - y_strings.shape[0]), (0, 0))),
        'mel_pitch': np.pad(y_mel_pitch, ((0, max_len - y_mel_pitch.shape[0]), (0, 0))),
        'vocals': np.pad(y_vocals, ((0, max_len - y_vocals.shape[0]), (0, 0)))
    }
    
    mixer = AudioMixer(padded_tracks, sr)

    with sd.OutputStream(callback=mixer.audio_callback, channels=2, samplerate=sr) as stream:
        print("Mixing audio. Press 'd', 's', 'm', 'v' and Enter to toggle tracks.")
        
        while True:
            cmd = input().strip().lower()
            if cmd == 'd':
                mixer.toggle_track('drums')
            elif cmd == 's':
                mixer.toggle_track('strings')
            elif cmd == 'm':
                mixer.toggle_track('mel_pitch')
            elif cmd == 'v':
                mixer.toggle_track('vocals')
            else:
                print("Invalid command. Use 'd', 's', 'm', or 'v'.")