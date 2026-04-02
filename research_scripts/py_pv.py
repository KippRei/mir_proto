import numpy as np
import sounddevice as sd
import librosa as l
import time

stretch_factor = float(input("Stretch Factor: "))
raw_audio, sr = l.load('..\\misc_mp3s\\Charli xcx - 360 (official lyric video).mp3', sr=48000)
# print(audio)
audio = raw_audio[2*sr:8*sr-1].copy()
# print(audio.size)
window_size = 2048
hann_window = np.hanning(window_size)
analysis_hop = int(window_size / 4)
synth_hop = int(analysis_hop * stretch_factor)
print(synth_hop)
num_frames = audio.size // analysis_hop
curr_frame = 0
synth_audio = np.zeros(int(audio.size * stretch_factor))
print(synth_audio.size)
# TODO: Only works when slowing down (stretch_factor >= 1)
while curr_frame < num_frames:
    frame = audio[curr_frame * analysis_hop: curr_frame * analysis_hop + window_size]
    if frame.size < window_size:
        break
    windowed_frame = hann_window * frame
    curr_fft = np.fft.fft(windowed_frame)
    curr_ifft = np.fft.ifft(curr_fft)
    output = hann_window * curr_ifft
    # print(curr_ifft)
    for idx, val in enumerate(curr_ifft):
        synth_audio[idx + (curr_frame * synth_hop)] += np.real(val).item()

    curr_frame += 1
new_audio = np.array(synth_audio)
# print(new_audio[0:10])

sr = 48000
sd.play(new_audio, sr)
try:
    # 3. Instead of sd.wait(), we poll the stream status
    while sd.get_stream().active:
        # A tiny sleep allows the Python interpreter to 
        # process signals like KeyboardInterrupt
        time.sleep(0.1) 
except KeyboardInterrupt:
    print("\nStopping audio...")
    sd.stop()  # Hard stop for the audio hardware
finally:
    print("Done.")