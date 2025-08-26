# Gets the 4 max points in autocorrelate, finds corresponding time, figures out BPM relative to other 4
# Finally use least squares method to find N (integer) BPM that's closest to all relative BPMs from step above (find integer BPM that is closest to all relative BPMs found above)

import librosa
import scipy
import numpy as np
import matplotlib.pyplot as plt
from math import inf
import pandas as pd

my_err = 0
lib_err = 0

song_list = pd.read_csv('bpm_data.csv')
for idx, row in song_list.iterrows():
    if idx > 10:
        break
    song_file_name = row['song']
    actual_bpm = row['bpm']

    y, sr = librosa.load(f"harmonixset/src/mp3s/{song_file_name}")
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    print(f'Librosa Estimated tempo: {tempo[0]:.1f} BPM')
    if round(tempo[0]) != actual_bpm:
        lib_err += 1

    NUM_OF_SEC = 8
    HOP_LEN = 512
    AC_SIZE = NUM_OF_SEC * sr // HOP_LEN
    FPS = sr / HOP_LEN
    odf = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LEN)
    max_corr_val = 0
    bpm_idx = -1
    bpm_clicks = ['45bpm', '46bpm', '47bpm', '48bpm', '49bpm', '50bpm', '51bpm', '52bpm', '53bpm', '54bpm', '55bpm', '56bpm', '57bpm', '58bpm', '59bpm', '60bpm', '61bpm', '62bpm', '63bpm', '64bpm', '65bpm', '66bpm', '67bpm', '68bpm', '69bpm', '70bpm', '71bpm', '72bpm', '73bpm', '74bpm', '75bpm', '76bpm', '77bpm', '78bpm', '79bpm', '80bpm', '81bpm', '82bpm', '83bpm', '84bpm', '85bpm', '86bpm', '87bpm', '88bpm', '89bpm', '90bpm', '91bpm', '92bpm', '93bpm', '94bpm', '95bpm', '96bpm', '97bpm', '98bpm', '99bpm', '100bpm', '101bpm', '102bpm', '103bpm', '104bpm', '105bpm', '106bpm', '107bpm', '108bpm', '109bpm', '110bpm', '111bpm', '112bpm', '113bpm', '114bpm', '115bpm', '116bpm', '117bpm', '118bpm', '119bpm', '120bpm', '121bpm', '122bpm', '123bpm', '124bpm', '125bpm', '126bpm', '127bpm', '128bpm', '129bpm', '130bpm', '131bpm', '132bpm', '133bpm', '134bpm', '135bpm', '136bpm', '137bpm', '138bpm', '139bpm', '140bpm', '141bpm', '142bpm', '143bpm', '144bpm', '145bpm', '146bpm', '147bpm', '148bpm', '149bpm', '150bpm', '151bpm', '152bpm', '153bpm', '154bpm', '155bpm', '156bpm', '157bpm', '158bpm', '159bpm', '160bpm', '161bpm', '162bpm']
    for idx, val in enumerate(bpm_clicks):
        ref_y, ref_sr = librosa.load(f"click_tracks/{val}.mp3")
        odf_ref = librosa.onset.onset_strength(y=ref_y, sr=ref_sr, hop_length=HOP_LEN)
        cc = scipy.signal.correlate(odf, odf_ref, mode='full')
        cc_sort = np.sort(cc)[::-1]
        cor_sum = cc_sort[0] + cc_sort[1] + cc_sort[2] + cc_sort[3]
        if cor_sum > max_corr_val:
            max_corr_val = cor_sum
            bpm_idx = idx

    actual_bpm_str = f"{actual_bpm}bpm"
    if bpm_clicks[bpm_idx] != actual_bpm_str:
            my_err += 1

    print(f"My Estimated BPM: {bpm_clicks[bpm_idx]}")
    print(f"Actual BPM: {actual_bpm}")
    print("\n")

print(f"Librosa errors = {lib_err}")
print(f"My errors = {my_err}")
    # fig, ax = plt.subplots()
    # ax.plot(ac)
    # ax.set(title='Auto-correlation', xlabel='Lag (frames)')
    # plt.show()