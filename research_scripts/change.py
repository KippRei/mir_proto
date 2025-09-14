import numpy as np
from pydub import AudioSegment
from scipy.signal import butter, lfilter

def apply_filter(segment, lowcut, highcut, btype, order):
    """Applies a SciPy filter to each channel of an AudioSegment and normalizes the output."""
    fs = segment.frame_rate
    
    # Generate the filter
    if btype == 'low':
        b, a = butter(order, lowcut / (0.5 * fs), btype=btype)
    elif btype == 'high':
        b, a = butter(order, highcut / (0.5 * fs), btype=btype)
    elif btype == 'band':
        b, a = butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype=btype)
    else:
        raise ValueError("Invalid filter type. Use 'low', 'high', or 'band'.")
    
    # Get audio data as a NumPy array and reshape for multichannel support
    samples = np.array(segment.get_array_of_samples())
    samples = samples.reshape(-1, segment.channels)
    
    # Create an array to store the filtered channels
    filtered_samples = np.zeros_like(samples, dtype=np.float64)
    
    # Loop through each channel and apply the filter separately
    for channel in range(segment.channels):
        y_channel = lfilter(b, a, samples[:, channel])
        filtered_samples[:, channel] = y_channel

    # Get the single max value from the entire filtered array for normalization
    max_val = np.max(np.abs(filtered_samples))
    if max_val == 0:
        # Avoid division by zero for silent tracks
        return AudioSegment.silent(duration=len(segment))
    
    # Normalize the filtered signal to prevent overflow/underflow
    y = (filtered_samples / max_val * (2**(segment.sample_width * 8 - 1) - 1)).astype(segment.array_type)
    
    # Reshape back to 1D before converting to bytes
    y = y.reshape(1, -1)[0]
    filtered_segment = segment._spawn(y.astype(segment.array_type))
    return filtered_segment

# Load the audio track
track = AudioSegment.from_mp3("stretched_carly_other.mp3")

# Define a higher filter order for wide bands
wide_band_order = 6

# Define a lower filter order for the narrow middle band
narrow_band_order = 2

# Define the frequency ranges
ranges = [(0, 160), (160, 170), (170, track.frame_rate / 2)]

# Separate the track and export each new track
for i, (low, high) in enumerate(ranges):
    # Determine the correct filter order based on the frequency range
    filter_order = wide_band_order
    if low == 160 and high == 170:
        filter_order = narrow_band_order
    
    # For the lowest range (0-160 Hz)
    if low == 0:
        filtered_track = apply_filter(track, high, 0, 'low', filter_order)
        filtered_track.export(f"track_low_{high}hz_order_{filter_order}.mp3", format="mp3")
        print(f"Exported low-pass track: 0 - {high} Hz with order {filter_order}")
    
    # For the highest range (170+ Hz)
    elif high == track.frame_rate / 2:
        filtered_track = apply_filter(track, 0, low, 'high', filter_order)
        filtered_track.export(f"track_high_{low}hz_order_{filter_order}.mp3", format="mp3")
        print(f"Exported high-pass track: {low}+ Hz with order {filter_order}")
    
    # For the middle range (160-170 Hz)
    else:
        filtered_track = apply_filter(track, low, high, 'band', filter_order)
        filtered_track.export(f"track_{low}hz_{high}hz_order_{filter_order}.mp3", format="mp3")
        print(f"Exported band-pass track: {low} - {high} Hz with order {filter_order}")