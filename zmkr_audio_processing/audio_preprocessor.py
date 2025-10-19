from zmkr_audio_processing import bpm_detection, split_into_stems, change_tempo, mp3_to_np
from PyQt6.QtCore import pyqtSignal, QObject
import librosa
from scipy import signal
import threading

# TODO: This must be in a separate thread, but playing while audio is preprocessing will likely cause stuttering due to CPU overload, ideas?
class AudioPreprocessor(QObject):
    new_audio_preprocessed = pyqtSignal()

    def __init__(self):
        super().__init__()

    def process_audio(self, file_name: str):
        audio_process_thread = threading.Thread(target=self.__process_audio, args=(file_name,), daemon=True)
        audio_process_thread.start()

    # TODO: ensure valid audio file type
    def __process_audio(self, file_name: str):
        orig_tempo, downbeats = bpm_detection.get_bpm(file_name)
        stems_to_process = split_into_stems.split(file_name)

        tempo_processed_folder = None
        if stems_to_process is not None:
            tempo_processed_folder, amt_tempo_changed = change_tempo.change_tempo(stems_to_process, orig_tempo)

        # TODO: Refactor this!
        # Trim silence from beginning of drum track to find good starting point (start_beat)
        # TODO: Figure out a way to get first beat using drums maybe? Currently, it is not consistent
        y, sr = librosa.load(f'{tempo_processed_folder}/drums.mp3', sr=48000)
        _, start_beat = librosa.effects.trim(y=y, top_db=15)
        print(f'Start beat: {start_beat}')

        closest_downbeat = 0
        min_dist_between = float('inf')
        for idx in range(len(downbeats)):
            # print(f'Stretch: {amt_to_stretch}')
            stretched_downbeat_sample_num = round((downbeats[idx] / amt_tempo_changed) * 48000)
            print(f'Downbeat Time: {downbeats[idx]}, Stretched Downbeat Sample: {stretched_downbeat_sample_num}')
            curr_dist_between = start_beat[0] - stretched_downbeat_sample_num
            if abs(curr_dist_between) < min_dist_between:
                print(f'Curr dist between: {curr_dist_between}')
                if curr_dist_between < -10000:
                    closest_downbeat = round((downbeats[idx + 1] / amt_tempo_changed) * 48000)
                else:
                    closest_downbeat = round((downbeats[idx] / amt_tempo_changed) * 48000)

                min_dist_between = abs(curr_dist_between)

        if tempo_processed_folder is not None:
            mp3_to_np.convert(tempo_processed_folder, closest_downbeat)

        self.new_audio_preprocessed.emit()
        print("Done preprocessing")