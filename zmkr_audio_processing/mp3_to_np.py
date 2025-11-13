import librosa
import numpy as np
import os

def convert(file_folder, start_beat):
    in_file_names = [
        '/vocals.mp3',
        '/drums.mp3',
        '/other.mp3',
        '/bass.mp3'
        ]
    out_file_names = [
        '/vocal',
        '/drum',
        '/melody',
        '/bass'
        ]

    # TODO: Hacky way of changing npy names to fit cases in drag and drop code (i.e. vocals.mp3->vocal.npy, drums.mp3->drum.npy, other.mp3->melody.npy)
    for idx, file_name in enumerate(in_file_names):
        fp = f"{file_folder}/{file_name}"

        # Get nparray of audio at 44.1 kHz in stereo
        y, sr = librosa.load(fp, sr=48000, mono=False)
        start_frame = start_beat
        # Get max length of column 2 of audio arrays for padding shorter songs (most samples = longest song)
        max_len = 4351282
        padded_y = None
        if y.shape[1] > max_len + start_frame:
            padded_y = np.transpose(y[:, start_frame:max_len + start_frame])
        # Pad each audio array with zeros to ensure all audio arrays same length
        else:
            # TODO: Need to ensure correct start frame when padding
            padded_y = np.transpose(np.pad(y, ((0, 0), (0, max_len - y.shape[1]))))

        dest_folder = f"{os.getcwd()}/preprocessed_audio/{file_folder.split('/')[-1]}"
        os.makedirs(dest_folder, exist_ok=True)
        np.save(f"{dest_folder}/{out_file_names[idx]}", padded_y)