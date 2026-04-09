import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from zmkr_audio_processing import bpm_detection, split_into_stems, change_tempo, mp3_to_np
from beat_this.inference import File2Beats
import librosa as l
import numpy as np
import torch
import math

audio_path = "..\\misc_mp3s\\Eminem - Houdini (Lyrics).mp3"
tempo, downbeats = bpm_detection.get_bpm(audio_path)
print(tempo)
file2beats = File2Beats(checkpoint_path="final0", device="cuda", dbn=False)


y, sr = l.load(audio_path, sr=22500)
spectrogram = l.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128)
spectro_db = l.power_to_db(spectrogram)
spectro_db = spectro_db.T
input_tensor = torch.from_numpy(spectro_db).unsqueeze(0).to("cuda")
samples_per_beat = (60 / tempo) * sr
samples_per_measure = 4 * samples_per_beat

# TODO: Maybe try with mel spectrogram instead of samples for better results
def get_similarity(idx):
    best_score = -math.inf
    phrase_idx = -1
    phrase = y[idx:int(idx+(4 * samples_per_measure))]
    for new_phrase_time in downbeats:
        new_phrase_idx = int(new_phrase_time * sr)
        if new_phrase_idx < idx:
            continue
        comparison_phrase = y[new_phrase_idx:int(new_phrase_idx+(4 * samples_per_measure))]
        score = sum(i[0] * i[1] for i in zip(phrase, comparison_phrase))
        if score > best_score:
            best_score = score
            phrase_idx = new_phrase_idx
    phrase_time = phrase_idx / sr
    return best_score, phrase_time

with torch.no_grad():
    outputs = file2beats.model(input_tensor)
    
downbeat_probs = outputs['downbeat'].cpu().squeeze()
normalized_probs = torch.sigmoid(downbeat_probs).numpy()
for idx, val in enumerate(normalized_probs):
    if val > 0.35:
        timestamp = (idx * 256) / sr
        sim_score, phrase_idx = get_similarity(idx * 256)
        print(f"Value: {val} .... Timestamp: {timestamp}")
        print(f"Best Phrase Match: {phrase_idx} .... Score: {sim_score}")