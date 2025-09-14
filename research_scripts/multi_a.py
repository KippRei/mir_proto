import librosa
import numpy as np
import math
import multiprocessing
from itertools import starmap

# The worker function can stay at the top level.
# Functions defined here are safe to be imported by child processes.
def compare_segments(filename, TIME_TO_CHECK, curr_time, MEASURE_LENGTH, NUM_MEASURES):
    """
    Loads two audio segments, computes their chroma features, and returns
    the DTW distance between them.
    """
    try:
        y1, sr1 = librosa.load(filename, offset=TIME_TO_CHECK, duration=MEASURE_LENGTH * NUM_MEASURES)
        chroma1 = librosa.feature.chroma_cqt(y=y1, sr=sr1)

        y2, sr2 = librosa.load(filename, offset=curr_time, duration=MEASURE_LENGTH * NUM_MEASURES)
        chroma2 = librosa.feature.chroma_cqt(y=y2, sr=sr2)

        D, wp = librosa.sequence.dtw(X=chroma1, Y=chroma2)
        dtw_distance = D[-1, -1]
        
        return dtw_distance, curr_time

    except Exception as e:
        print(f"An error occurred for time {curr_time}: {e}")
        return float('inf'), curr_time

# --- Main Script ---
# All of the main execution logic must be inside this block.
if __name__ == '__main__':
    filename = 'twofriends.mp3'
    y, sr = librosa.load(filename)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    MEASURE_LENGTH = beat_times[4] - beat_times[0]
    NUM_MEASURES = 8

    beat_times_iter = iter(beat_times)
    TIME_TO_CHECK = next(beat_times_iter, None)

    while TIME_TO_CHECK is not None:
        if TIME_TO_CHECK < 8:
            TIME_TO_CHECK = next(beat_times_iter)
            continue

        closest_match = math.inf
        closest_time = 0

        tasks = [(filename, TIME_TO_CHECK, time2, MEASURE_LENGTH, NUM_MEASURES) for time2 in beat_times if time2 > TIME_TO_CHECK + (MEASURE_LENGTH * NUM_MEASURES)]

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.starmap(compare_segments, tasks)
            
        for dtw_distance, curr_time in results:
            if dtw_distance < closest_match:
                closest_match = dtw_distance
                closest_time = curr_time

        if closest_time <= TIME_TO_CHECK + (MEASURE_LENGTH * NUM_MEASURES):
            NUM_MEASURES *= 2
        else:
            print(f"{TIME_TO_CHECK}, {closest_time}")
            NUM_MEASURES = 8
            TIME_TO_CHECK = next(beat_times_iter)