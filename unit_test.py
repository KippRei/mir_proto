import pytest
from zmkr_audio_processing import bpm_detection as bp_det
from rust_audio_manager import AudioPlayer
from midi_manager import MIDIManager
from midi_controller import MIDIController
import mido


class TestBpm:
    def test_correct_bpm(self):
        detected_bpm = bp_det.get_bpm("C:\\Users\\jappa\\Repos\\sr_proto\\mir_1\\misc_mp3s\\Andrea Botez - SOS.mp3")[0]
        assert detected_bpm == pytest.approx(136.18835293309988)

    def test_incorrect_bpm(self):
        detected_bpm = bp_det.get_bpm("C:\\Users\\jappa\\Repos\\sr_proto\\mir_1\\misc_mp3s\\apollo.mp3")[0]
        assert detected_bpm != pytest.approx(136.18835293309988)

class TestMidiInput:
    audio_manager = AudioPlayer()
    midi_manager = MIDIManager()
    midi_controller = MIDIController(audio_manager, midi_manager)

    def test_good_input(self):
        good_message = mido.Message('note_on', channel=0, note=36, velocity=127)
        assert self.midi_controller.process_msg(good_message) == True

    def test_bad_input(self):
        bad_message = mido.Message('note_on', channel=0, note=127, velocity=127)
        assert self.midi_controller.process_msg(bad_message) == False
