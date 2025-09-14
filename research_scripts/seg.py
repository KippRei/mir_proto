import librosa
import numpy as np
import matplotlib.pyplot as plt

FILE_NAME = 'twofriends.mp3'
# 1. Load an example audio file
y, sr = librosa.load(FILE_NAME)

# 2. Extract a feature representation for structural analysis.
# Chroma features (pitch content) are often a good choice for this.
# CQT (Constant-Q Transform) is used here for better frequency resolution.
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
mfcc = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=20, n_fft=4096, n_mels=256)

# 3. Build a recurrence matrix from the features.
# This matrix shows how similar each point in time is to every other point.
# A bright spot means the music at those two points in time is similar.
rec_matrix = librosa.segment.recurrence_matrix(
    mfcc,
    mode='affinity',
    metric='cosine'  # Use cosine similarity, which is robust to loudness changes
)

# 4. Use the recurrence matrix to identify structural segments.
# We'll use agglomerative clustering to find the boundaries.
# 'k' is the number of segments we want to find. Let's assume we want 5 sections.
n_segments = 5
bounds_frames = librosa.segment.agglomerative(rec_matrix, k=n_segments)

# 5. Convert the frame indices to timestamps for a more intuitive output.
bounds_times = librosa.frames_to_time(bounds_frames, sr=sr)

# 6. Print the detected segment boundaries
print(f"Detected {len(bounds_times)} segment boundaries:")
for time in bounds_times:
    print(f"- {time:.2f} seconds")

# 7. Visualize the recurrence matrix and the detected boundaries
fig, ax = plt.subplots()
img = librosa.display.specshow(rec_matrix, x_axis='time', y_axis='time', ax=ax, sr=sr)
ax.set(title='Recurrence Matrix with Detected Segment Boundaries')

# Overlay the segment boundaries on the plot
for t in bounds_times:
    ax.axvline(t, color='r', linestyle='--', linewidth=2)
    ax.axhline(t, color='r', linestyle='--', linewidth=2)

fig.colorbar(img, ax=ax, label='Similarity (cosine)')
# plt.show()