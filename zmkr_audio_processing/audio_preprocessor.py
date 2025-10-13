from zmkr_audio_processing import bpm_detection, split_into_stems, change_tempo, mp3_to_np
from PyQt6.QtCore import pyqtSignal, QObject
import librosa
from scipy import signal

# TODO: This must be in a separate thread!!
class AudioPreprocessor(QObject):
    new_audio_preprocessed = pyqtSignal()

    def __init__(self, audio_manager):
        super().__init__()
        self.audio_manager = audio_manager

    # TODO: ensure valid audio file type
    def process_audio(self, file_name: str):
        orig_tempo, downbeats = bpm_detection.get_bpm(file_name)
        stems_to_process = split_into_stems.split(file_name)

        tempo_processed_folder = None
        if stems_to_process is not None:
            tempo_processed_folder, amt_to_stretch = change_tempo.change_tempo(stems_to_process, orig_tempo)

        # Trim silence from beginning of drum track to find good starting point (start_beat)
        # TODO: Figure out a way to get first beat using drums maybe? Currently, it is not consistent
        y, sr = librosa.load(f'{tempo_processed_folder}/drums.mp3', sr=44100)
        _, start_beat = librosa.effects.trim(y=y, top_db=20)

        closest_downbeat = 0
        dist_to_downbeat = float('inf')
        for downbeat in downbeats:
            print(f'Stretch: {amt_to_stretch}')
            stretched_downbeat_sample_num = round((downbeat / amt_to_stretch) * 44100)
            print(f'Downbeat Time: {downbeat}, Stretched Downbeat Sample: {stretched_downbeat_sample_num}')
            if abs(start_beat[0] - stretched_downbeat_sample_num) < dist_to_downbeat:
                closest_downbeat = stretched_downbeat_sample_num
                dist_to_downbeat = abs(start_beat[0] - stretched_downbeat_sample_num)
                print(f'Closest downbeat: {closest_downbeat}, Dist: {dist_to_downbeat}')

        if tempo_processed_folder is not None:
            mp3_to_np.convert(tempo_processed_folder, closest_downbeat)

        self.audio_manager.load_preprocessed_songs()
        self.new_audio_preprocessed.emit()
        print("Done preprocessing")