from pydub import AudioSegment
import numpy as np
import os

def generate_click_track(bpm, duration_sec=30):
    # Set the sample rate and frequency for the click sound
    sample_rate = 44100
    click_freq = 1000  # A high-pitched click

    # Calculate the period of the clicks in samples
    # (60 sec/min) / (BPM) * (samples/sec)
    period_samples = int((60 / bpm) * sample_rate)

  # Generate a short, sharp pulse (a single high-amplitude sample)
    pulse_length_samples = 50  # Adjust for a shorter or longer click
    click_wave = np.zeros(pulse_length_samples, dtype=np.int16)
    click_wave[0] = 32767  # Max amplitude for a sharp spike

    # Create a silent audio segment for the full duration
    track_length_samples = duration_sec * sample_rate
    click_track = np.zeros(track_length_samples, dtype=np.int16)

    # Place the clicks at regular intervals
    for i in range(0, track_length_samples, period_samples):
        end_idx = i + len(click_wave)
        if end_idx < track_length_samples:
            click_track[i:end_idx] = click_wave

    # Convert the numpy array to an AudioSegment
    audio_segment = AudioSegment(
        click_track.tobytes(),
        frame_rate=sample_rate,
        sample_width=click_track.dtype.itemsize,
        channels=1
    )
    
    # Export the audio to an MP3 file
    filename = f"{bpm}bpm.mp3"
    audio_segment.export(filename, format="mp3")
    print(f"Generated {filename}")

# Create a folder to store the click tracks
if not os.path.exists("click_tracks"):
    os.makedirs("click_tracks")
os.chdir("click_tracks")

# Loop through the desired BPM range
for bpm in range(45, 181):
    generate_click_track(bpm)