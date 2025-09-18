import librosa
import numpy as np

# File paths of stems for testing (will auto-load when program starts)
fp = 'doechii_maybe/doechii_maybe_drums.mp3'
fp2 = 'doechii_maybe/doechii_maybe_strings.mp3'
fp3 = 'doechii_maybe/doechii_maybe_no_one_mel_pitch.mp3'
fp4 = 'doechii_maybe/doechii_maybe_vocals.mp3'

# Get np array of audio at 44.1 kHz in stereo
y, sr = librosa.load(fp, sr=44100, mono=False)
y1, sr = librosa.load(fp2, sr=44100, mono=False)
y2, sr = librosa.load(fp3, sr=44100, mono=False)
y3, sr = librosa.load(fp4, sr=44100, mono=False)

# Get max length of column 2 of audio arrays for padding shorter songs (most samples = longest song)
max_len = max(y.shape[1], y1.shape[1], y2.shape[1], y3.shape[1])

# Pad each audio array with zeros to ensure all audio arrays same length
padded_y = np.transpose(np.pad(y, ((0, 0), (0, max_len - y.shape[1]))))
padded_y1 = np.transpose(np.pad(y1, ((0, 0), (0, max_len - y1.shape[1]))))
padded_y2 = np.transpose(np.pad(y2, ((0, 0), (0, max_len - y2.shape[1]))))
padded_y3 = np.transpose(np.pad(y3, ((0, 0), (0, max_len - y3.shape[1]))))
print(padded_y.shape)

tr = [padded_y, padded_y1, padded_y2, padded_y3]
for idx, t in enumerate(tr):
    np.save(f"C:/Users/jappa/Repos/senior_project/preprocessed_audio/{idx}", t)