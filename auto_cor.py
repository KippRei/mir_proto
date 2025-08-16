# Gets the 4 max points in autocorrelate, finds corresponding time, figures out BPM relative to other 4
# Finally use least squares method to find N (integer) BPM that's closest to all relative BPMs from step above (find integer BPM that is closest to all relative BPMs found above)

import librosa
import matplotlib.pyplot as plt
from math import inf

y, sr = librosa.load("up_from_the_bottom.mp3")
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
print(f'Estimated tempo: {tempo[0]:.2f} BPM')

NUM_OF_SEC = 8
HOP_LEN = 512
AC_SIZE = NUM_OF_SEC * sr // HOP_LEN
FPS = sr / HOP_LEN
odf = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LEN)
ac = librosa.autocorrelate(odf, max_size=AC_SIZE)

v1, v2, v3, v4 = 0, 0, 0, 0
v1_idx, v2_idx, v3_idx, v4_idx = -1, -1, -1, -1
for idx, val in enumerate(ac):
    if val > v1:
        v1 = val
        v1_idx = idx
    elif val > v2:
        v4 = v3
        v4_idx = v3_idx
        v3 = v2
        v3 = v2_idx
        v2 = val
        v2_idx = idx
    elif val > v3:
        v4 = v3
        v4_idx = v3_idx
        v3 = val
        v3_idx = idx
    elif val > v4:
        v4 = val
        v4_idx = idx

index_list = [v1_idx, v2_idx, v3_idx, v4_idx]
index_list.sort()
time_diff = []
for i in index_list:
    for j in index_list:
        if j - i <= 0:
            continue
        time_elapsed = (j - i) / FPS
        time_diff.append(time_elapsed)

min_err = inf
bpm_est = 0
for N in range(60,180):
    least_sq_err = 0
    for n in time_diff:
        non_normalized_bpm = 60 / n
        m = round(N / non_normalized_bpm)
        least_sq_err += pow((non_normalized_bpm * m - N), 2)
    if least_sq_err < min_err:
        min_err = least_sq_err
        bpm_est = N

print(bpm_est)

# fig, ax = plt.subplots()
# ax.plot(ac)
# ax.set(title='Auto-correlation', xlabel='Lag (frames)')
# plt.show()