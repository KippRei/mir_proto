import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def analyze_song_structure(file_path):
    """
    Analyzes a song's structure by detecting section boundaries.
    This version is updated for modern librosa (v0.10.x and later)
    and corrects the 2D convolution for the novelty curve.
    """
    
    # --- 1. Load the audio file ---
    print(f"Loading audio from: {file_path}")
    # Using mono=True ensures a single channel, which is standard for most analysis
    y, sr = librosa.load(file_path, duration=60, mono=True)
    
    # --- 2. Extract Features: Chroma ---
    print("Extracting chroma features...")
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # Optional: Beat-synchronize the features for a cleaner matrix.
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    chroma_synced = librosa.util.sync(chroma, beats)
    
    # --- 3. Build the Self-Similarity Matrix ---
    print("Computing self-similarity matrix...")
    R = librosa.segment.recurrence_matrix(
        chroma_synced,
        mode='affinity',
        metric='cosine',
        k=10, 
        width=3, 
        sym=True 
    )
    
    # --- 4. Calculate the Novelty Curve (Updated & Corrected Method) ---
    print("Calculating novelty curve...")
    
    # Define a simple checkerboard-like kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])
    
    # Use scipy's 2D convolution. 'mode=same' is what we want here,
    # as it returns an array of the same size as the input.
    novelty_matrix = convolve2d(R, kernel, mode='same', boundary='symm')
    
    # The final novelty curve is a diagonal slice of the convolved matrix
    novelty_curve = np.diag(novelty_matrix)
    
    # --- 5. Find Section Boundaries and Convert to Timestamps ---
    # We use peak picking on the novelty curve to find the boundary frames.
    boundary_frames = librosa.util.peak_pick(
        novelty_curve,
        pre_max=3,
        post_max=3,
        pre_avg=3,
        post_avg=3,
        delta=0.5,
        wait=3
    )
    
    # Convert beat frames to time in seconds
    boundary_times = librosa.frames_to_time(beats[boundary_frames], sr=sr)
    
    print("\nDetected section boundaries (in seconds):")
    for time in boundary_times:
        print(f" - {time:.2f}s")

    # --- 6. Visualization (for understanding) ---
    plt.figure(figsize=(12, 8))

    # Plot the Self-Similarity Matrix
    plt.subplot(2, 1, 1)
    librosa.display.specshow(R, y_axis='time', x_axis='time', sr=sr, hop_length=librosa.get_duration(y=y)/R.shape[0])
    plt.title('Self-Similarity Matrix')
    plt.vlines(boundary_times, 0, R.shape[0], color='r', linestyle='--', label='Section Boundaries')
    plt.hlines(boundary_times, 0, R.shape[0], color='r', linestyle='--')
    plt.legend()

    # Plot the Novelty Curve with boundaries
    plt.subplot(2, 1, 2)
    plt.plot(librosa.times_like(novelty_curve, sr=sr, hop_length=librosa.get_duration(y=y)/novelty_curve.shape[0]), novelty_curve, label='Novelty Curve')
    plt.vlines(boundary_times, 0, novelty_curve.max(), color='r', linestyle='--', label='Section Boundaries')
    plt.title('Novelty Curve')
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Run the function with a sample audio file ---
try:
    file_path = 'twofriends.mp3'
    analyze_song_structure(file_path)
except FileNotFoundError:
    print("Librosa example file not found. Please provide a local path to an audio file.")
    # Example for local file:
    # file_path = "path/to/your/song.wav"
    # analyze_song_structure(file_path)