import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Load an audio file
y, sr = librosa.load('doechii_maybe/doechii_maybe_strings.mp3')

# Compute the STFT
D = librosa.stft(y, n_fft=8192)

# Convert the amplitude to decibels for visualization
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# Create a figure and axes for the plot
fig, ax = plt.subplots(figsize=(10, 5))

# Display the spectrogram
img = librosa.display.specshow(S_db,
                               sr=sr,
                               x_axis='time',
                               y_axis='log',
                               ax=ax)

# Add a colorbar and title for clarity
fig.colorbar(img, ax=ax, format="%+2.f dB")
ax.set(title='Spectrogram with Natural Note Labels')

# List of natural notes and their corresponding MIDI numbers
natural_notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']

# Create a list of MIDI notes for all octaves from C1 to B7
midi_notes = []
for octave in range(1, 8):
    for note in natural_notes:
        midi_notes.append(note + str(octave))

# Convert the MIDI notes to their fundamental frequencies (in Hz)
note_frequencies = librosa.note_to_hz(midi_notes)
# Convert the MIDI notes to labels for the y-axis
note_labels = midi_notes

# Set the y-axis ticks to the natural note frequencies
ax.set_yticks(note_frequencies)
# Set the y-axis tick labels to the natural note names
ax.set_yticklabels(note_labels)

# Show the plot
plt.show()