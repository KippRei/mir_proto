from zmkr_audio_processing import bpm_detection, split_into_stems, change_tempo

class AudioPreprocessor():
    def __init__(self):
        pass

    # TODO: ensure valid audio file type
    def process_audio(self, file_name: str):
        tempo = bpm_detection.get_bpm(file_name)
        stems_to_process = split_into_stems.split(file_name)
        stem_types = [
            'bass',
            'drums',
            'other',
            'vocals'
        ]
        if stems_to_process is not None:
            for type in stem_types:
                fp = stems_to_process + "/" + type + ".mp3"
                change_tempo(fp)