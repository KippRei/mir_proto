import numpy as np
import sounddevice as sd
import librosa as l
import time
import math

stretch_factor = float(input("Stretch Factor: "))
raw_audio, sr = l.load('..\\stems_demucs\\htdemucs\\Charli xcx - 360 (official lyric video)\\drums.mp3', sr=48000)

# print(audio)
audio = raw_audio[32*sr:42*sr].copy()
# print(audio.size)
window_size = 2048
hann_window = np.hanning(window_size)
analysis_hop = window_size // 4
synth_hop = int(analysis_hop * stretch_factor)
print(synth_hop)
num_frames = audio.size // analysis_hop
print(f'Num frames: {num_frames}')
curr_frame = 0
synth_audio = np.zeros(int(num_frames * synth_hop) + 2048)
phase_acc = np.zeros(int(window_size))
print(synth_audio.size)

# Phase accumulator stuff
phi_target = np.zeros(int(window_size))
phi_prev = np.zeros(int(window_size))

for idx in range(window_size):
    phi_target[idx] = (2 * math.pi * idx * analysis_hop) / window_size


def analyze(curr_fft):
    analyzed_fft = np.zeros(window_size, dtype=np.complex128)
    
    for idx,val in enumerate(curr_fft):
        mag_curr = np.abs(val)
        phi_curr = np.angle(val)
        phi_actual =  phi_curr - phi_prev[idx]
        phi_deviation = phi_actual - phi_target[idx]
        # TODO: Need to go deeper on wrapping
        wrapped_dev = ((phi_deviation + math.pi) % (2 * math.pi)) - math.pi
        phi_adv = ((2 * math.pi * idx * synth_hop) / window_size) + wrapped_dev
        phase_acc[idx] += phi_adv
        phase_acc[idx] = (phase_acc[idx] + math.pi) % (2 * math.pi) - math.pi
        new_val = mag_curr * np.exp(1j * phase_acc[idx])
        analyzed_fft[idx] = new_val
        phi_prev[idx] = phi_curr
    
    return analyzed_fft

while curr_frame < num_frames:
    frame = audio[curr_frame * analysis_hop: curr_frame * analysis_hop + window_size]
    if frame.size < window_size:
        break
    windowed_frame = hann_window * frame
    curr_fft = np.fft.fft(windowed_frame)
    analyzed_fft = analyze(curr_fft)
    curr_ifft = np.fft.ifft(analyzed_fft)
    output = hann_window * curr_ifft
    # print(curr_ifft)
    for idx, val in enumerate(output):
        synth_audio[idx + math.floor(curr_frame * synth_hop)] += np.real(val).item()
    curr_frame += 1


new_audio = np.array(synth_audio)
# print(new_audio[0:10])

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