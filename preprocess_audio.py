import librosa
import numpy as np

file_folder = './doechii_maybe'
file_names = [
    'doechii_sos_bass.mp3',
    'doechii_sos_drum.mp3',
    'doechii_sos_vocal.mp3',
    'doechii_sos_melody.mp3',
    'doechii_maybe_drum.mp3',
    'doechii_maybe_vocal.mp3',
    'doechii_maybe_strings.mp3'
]

for file_name in file_names:
    fp = f"{file_folder}/{file_name}"

    # Get nparray of audio at 48 kHz in stereo
    y, sr = librosa.load(fp, sr=48000, mono=False)

    # Get max length of column 2 of audio arrays for padding shorter songs (most samples = longest song)
    max_len = 4351282 * 2
    padded_y = None
    if y.shape[1] > max_len:
        padded_y = np.transpose(y[:max_len])
    # Pad each audio array with zeros to ensure all audio arrays same length
    else:
        padded_y = np.transpose(np.pad(y, ((0, 0), (0, max_len - y.shape[1]))))

    np.save(f"C:/Users/jappa/Repos/senior_project/preprocessed_audio/{file_name.split('.')[0]}", padded_y)