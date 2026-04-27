import pytest
from zmkr_audio_processing import audio_preprocessor as ap
from rust_audio_manager import AudioPlayer
from midi_manager import MIDIManager
from midi_controller import MIDIController
import mido


class TestDownbeat:
    def test_correct_downbeat(self):
        audio_processor = ap.AudioPreprocessor()
        detected_downbeat = audio_processor.get_downbeat([97000], [1.0, 2.0, 3.0], 1.0)
        assert detected_downbeat == 96000

    def test_incorrect_downbeat(self):
        audio_processor = ap.AudioPreprocessor()
        detected_downbeat = audio_processor.get_downbeat([97000], [1.0, 2.0, 3.0], 1.0)
        assert detected_downbeat != 97000

class TestMidiControlInput:
    audio_manager = AudioPlayer()
    midi_manager = MIDIManager()
    midi_controller = MIDIController(audio_manager, midi_manager)

    def test_good_input(self):
        good_message = mido.Message('control_change', channel=0, control=14, value=127)
        assert self.midi_controller.process_msg(good_message) == True

    def test_bad_input(self):
        bad_message = mido.Message('control_change', channel=0, control=127, value=127)
        assert self.midi_controller.process_msg(bad_message) == False
