# Librosa Example
# # Beat tracking example from librosa docs
import librosa
import numpy
import math

filename = 'twofriends.mp3'
y, sr = librosa.load(filename)

tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

MEASURE_LENGTH = beat_times[4] - beat_times[0]
NUM_MEASURES = 1

beat_times_iter = iter(beat_times)
TIME_TO_CHECK = next(beat_times_iter, None)

while TIME_TO_CHECK is not None:
        # TESTING: so we can start around the beginning of verse 1
        if TIME_TO_CHECK < 8:
                TIME_TO_CHECK = next(beat_times_iter)
                continue

        closest_match = math.inf
        closest_time = 0

        for time2 in beat_times:
                curr_time = time2
                if curr_time <= TIME_TO_CHECK + (MEASURE_LENGTH * NUM_MEASURES):
                        continue
                y1, sr1 = librosa.load(filename, offset=TIME_TO_CHECK, duration=MEASURE_LENGTH * NUM_MEASURES)
                mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=20, n_fft=4096, n_mels=256)
                y2, sr2 = librosa.load(filename, offset=curr_time, duration=MEASURE_LENGTH * NUM_MEASURES)
                mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=20, n_fft=4096, n_mels=256)
                D, wp = librosa.sequence.dtw(X=mfcc1, Y=mfcc2)
                dtw_distance = D[-1, -1]
                if dtw_distance < closest_match:
                        closest_match = dtw_distance
                        closest_time = curr_time

        if closest_time <= TIME_TO_CHECK + (MEASURE_LENGTH * NUM_MEASURES):
                NUM_MEASURES *= 2
        else:
                print(f"{TIME_TO_CHECK}, {closest_time}")
                NUM_MEASURES = 1
                TIME_TO_CHECK = next(beat_times_iter)