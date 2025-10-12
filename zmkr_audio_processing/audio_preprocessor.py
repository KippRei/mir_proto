from zmkr_audio_processing import bpm_detection, split_into_stems, change_tempo, mp3_to_np
from PyQt6.QtCore import pyqtSignal, QObject

# TODO: This must be in a separate thread!!
class AudioPreprocessor(QObject):
    new_audio_preprocessed = pyqtSignal()

    def __init__(self, audio_manager):
        super().__init__()
        self.audio_manager = audio_manager

    # TODO: ensure valid audio file type
    def process_audio(self, file_name: str):
        orig_tempo, orig_start_beat = bpm_detection.get_bpm(file_name)
        stems_to_process = split_into_stems.split(file_name)

        tempo_processed_folder = None
        if stems_to_process is not None:
            tempo_processed_folder, new_start_beat = change_tempo.change_tempo(stems_to_process, orig_tempo, orig_start_beat)

        # TODO: Figure out a way to get first beat using drums maybe? Currently, it is not consistent
        new_tempo, new_start_beat = bpm_detection.get_bpm(f'{tempo_processed_folder}/drums.mp3')
        # print(f'{new_tempo}: {new_start_beat}')
        if tempo_processed_folder is not None:
            mp3_to_np.convert(tempo_processed_folder, new_start_beat)

        self.audio_manager.load_preprocessed_songs()
        self.new_audio_preprocessed.emit()