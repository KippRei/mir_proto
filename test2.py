#!/usr/bin/env python
 
import numpy as np
from pydub import AudioSegment
import aubio
 
# The path to your song file
song_file_path = "twofriends.mp3"
 
# --- Audio File Loading and Preparation ---
 
print(f"Loading audio file: {song_file_path}")
 
try:
    # Load the audio file using pydub
    audio = AudioSegment.from_file(song_file_path)
 
    # Resample to mono if needed
    audio = audio.set_channels(1)
    # Get the actual samplerate from the loaded audio file
    samplerate = audio.frame_rate
 
    # Convert pydub audio data to a float32 NumPy array
    # Normalize the samples to the range [-1, 1]
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples /= 2**15 # Assuming 16-bit audio from pydub by default
 
except FileNotFoundError:
    print(f"Error: The file '{song_file_path}' was not found.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the audio file: {e}")
    exit()
 
# --- Aubio Configuration (Corrected) ---
 
# Define the buffer and hop sizes.
# win_size is the window size for analysis.
# hop_size is the step size between windows.
win_size = 1024
hop_size = 512
 
# Now, initialize the aubio tempo detection object with the correct samplerate.
tempo_detector = aubio.tempo("default", win_size, hop_size, samplerate)
 
# --- Processing the Audio ---
 
beats = [] # A list to store the time of each detected beat
total_frames = len(samples)
current_frame = 0
 
print("Processing audio for beats...")
 
while current_frame < total_frames:
    # Get a buffer of audio data
    end_frame = min(current_frame + hop_size, total_frames)
    buffer = samples[current_frame:end_frame]
 
    # Pad the buffer with zeros if it's too short at the end
    if len(buffer) < hop_size:
        buffer = np.pad(buffer, (0, hop_size - len(buffer)), 'constant')
 
    # Feed the buffer to the tempo detector
    is_beat = tempo_detector(buffer)
 
    if is_beat:
        # If a beat is detected, get the time in seconds and add it to our list
        beat_time = tempo_detector.get_last_s()
        beats.append(beat_time)
 
    # Move to the next chunk
    current_frame += hop_size
 
# --- Final BPM Calculation ---
 
print(f"Total beats detected: {len(beats)}")
 
if len(beats) > 1:
    # Calculate the time intervals between consecutive beats
    beat_intervals = np.diff(beats)
 
    # Convert beat intervals (in seconds) to instantaneous BPM values
    bpms = 60. / beat_intervals
 
    # The median is more robust to outliers than the mean
    final_bpm = np.median(bpms)
 
    # `aubio` also provides its own BPM estimate
    final_bpm_aubio = tempo_detector.get_bpm()
 
    print(f"Median BPM from beat intervals: {final_bpm:.2f}")
    print(f"Aubio's final BPM estimate: {final_bpm_aubio:.2f}")
else:
    print("Not enough beats were detected to calculate a reliable BPM.")