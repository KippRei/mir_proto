import numpy as np
from pydub import AudioSegment
from scipy.signal import butter, lfilter

def apply_band_pass_filter(segment, lowcut, highcut, order):
    """Applies a SciPy band-pass filter to an AudioSegment."""
    fs = segment.frame_rate
    nyquist = 0.5 * fs
    
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')
    
    samples = np.array(segment.get_array_of_samples())
    samples = samples.reshape(-1, segment.channels)
    
    filtered_samples = np.zeros_like(samples, dtype=np.float64)
    
    for channel in range(segment.channels):
        y_channel = lfilter(b, a, samples[:, channel])
        filtered_samples[:, channel] = y_channel

    max_val = np.max(np.abs(filtered_samples))
    if max_val == 0:
        return AudioSegment.silent(duration=len(segment))
    
    y = (filtered_samples / max_val * (2**(segment.sample_width * 8 - 1) - 1)).astype(segment.array_type)
    y = y.reshape(1, -1)[0]
    return segment._spawn(y)

# Load the audio track
track = AudioSegment.from_mp3("stretched_carly_other.mp3")

# Use a high-order filter for a very steep roll-off
filter_order = 12

# Define the target notes and a narrow bandwidth
notes_to_isolate = {
    "E3": 164.81,
    "E4": 329.62,
    "E5": 659.24
}
bandwidth_hz = 4

# Create a list to hold the isolated tracks
isolated_tracks_list = []

# Loop through each note and create a separate filtered track
for note_name, target_hz in notes_to_isolate.items():
    lowcut = target_hz - (bandwidth_hz / 2)
    highcut = target_hz + (bandwidth_hz / 2)

    isolated_track = apply_band_pass_filter(track, lowcut, highcut, filter_order)
    
    # Export the isolated track
    filename = f"isolated_{note_name}.mp3"
    isolated_track.export(filename, format="mp3")
    print(f"Exported track with isolated frequency range: {lowcut} Hz to {highcut} Hz for note {note_name}")
    
    # Append the isolated track to the list
    isolated_tracks_list.append(isolated_track)

# Combine the isolated tracks from the list using sum()
if isolated_tracks_list:
    combined_track = sum(isolated_tracks_list)
    # Export the final combined track
    combined_track.export("combined_octaves.mp3", format="mp3")
    print("Exported final combined track: combined_octaves.mp3")
else:
    print("No tracks were isolated to combine.")