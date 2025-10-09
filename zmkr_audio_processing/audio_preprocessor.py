from zmkr_audio_processing import bpm_detection, split_into_stems, change_tempo

class AudioPreprocessor():
    def __init__(self):
        pass

    # TODO: ensure valid audio file type
    def process_audio(self, file_name: str):
        tempo = bpm_detection.get_bpm(file_name)
        stems_to_process = split_into_stems.split(file_name)

        if stems_to_process is not None:
            change_tempo.change_tempo(stems_to_process, tempo)