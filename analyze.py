# Librosa Example
# # Beat tracking example from librosa docs
import librosa
import numpy
import math
# import cupy as cp

filename = 'separated/htdemucs/twofriends/other.mp3'
y, sr = librosa.load(filename)

tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

MEASURE_LENGTH = beat_times[4] - beat_times[0]

beat_times_iter = iter(beat_times)
TIME_TO_CHECK = next(beat_times_iter, None)

BEATS_PER_MEAS = 4

while TIME_TO_CHECK is not None:
        # TESTING: so we can start around the beginning of verse 1
        if TIME_TO_CHECK < 8:
                TIME_TO_CHECK = next(beat_times_iter)
                continue

        phrase_length = 1
        closest_match = math.inf
        closest_time = 0
        y1, sr1 = librosa.load(filename, offset=TIME_TO_CHECK, duration=MEASURE_LENGTH)
        chroma1 = librosa.feature.chroma_cqt(y=y1, sr=sr1)

        for beat_num, time2 in enumerate(beat_times):
                curr_time = time2
                if curr_time <= TIME_TO_CHECK + (MEASURE_LENGTH) or (beat_num % BEATS_PER_MEAS) + 1 != 1:
                        continue

                y2, sr2 = librosa.load(filename, offset=curr_time, duration=MEASURE_LENGTH)
                chroma2 = librosa.feature.chroma_cqt(y=y2, sr=sr2)
                
                # # TODO: GPU DTW
                # chroma1_gpu = cp.asarray(chroma1)
                # chroma2_gpu = cp.asarray(chroma2)

                #CPU DTW
                D, wp = librosa.sequence.dtw(X=chroma1, Y=chroma2)
                dtw_distance = D[-1, -1]
                
                if dtw_distance < closest_match:
                        closest_match = dtw_distance
                        closest_time = curr_time

        if closest_time <= TIME_TO_CHECK + MEASURE_LENGTH:
                phrase_length += 1
                TIME_TO_CHECK = next(beat_times_iter)
        else:
                print(f"{TIME_TO_CHECK}, {closest_time}, {phrase_length} bars")
                TIME_TO_CHECK = next(beat_times_iter)