# # Librosa Example
# # # Beat tracking example
# import librosa
# import numpy

# # # 1. Get the file path to an included audio example
# filename = 'no_one_else.mp3'

# # 2. Load the audio as a waveform `y`
# #    Store the sampling rate as `sr`
# y, sr = librosa.load(filename)

# # 3. Run the default beat tracker
# tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

# print('Estimated tempo: {:.2f} beats per minute'.format(tempo[0]))

# 4. Convert the frame indices of beat events into timestamps
# beat_times = librosa.frames_to_time(beat_frames, sr=sr)

# for idx, b in enumerate(beat_times):
#     print(str(idx) + ": " + str(b))


# Pyrubberband example
# import soundfile as sf
# import pyrubberband as pyrb

# # tracks = ['bass', 'other', 'vocals']
# # for i in tracks:
# #     y, sr = sf.read("separated/htdemucs/twofriends/{}.mp3".format(i))
# #     output_file = "pitched_{}.mp3".format(i)
# #     # Play back at double speed
# #     # y_stretch = pyrb.time_stretch(y, sr, 2.0)
# #     # Play back two semi-tones higher
# #     y_shift = pyrb.pitch_shift(y, sr, 2)
# #     sf.write(output_file, y_shift, sr, format='MP3')

# y, sr = sf.read("twofriends.mp3")
# output_file = "stretched_friends.mp3"
# # Play back at double speed
# y_stretch = pyrb.time_stretch(y, sr, 2.0)
# # Play back two semi-tones higher
# # y_shift = pyrb.pitch_shift(y, sr, 2)
# sf.write(output_file, y_stretch, sr, format='MP3')

# BPM change
import soundfile as sf
import pyrubberband as pyrb
import librosa

bpm_filename = 'doechii.mp3'
in_filename = 'separated/htdemucs/no_one_else/other.mp3'
out_filename = 'stretched_pitched_no_one_other.mp3'

y, sr = librosa.load(bpm_filename)
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
orig_tempo = tempo[0] - 1
for t in tempo:
    print('Estimated original tempo: {:.2f} beats per minute'.format(t))

new_tempo = float(input('Tempo to play song: '))
amt_to_stretch = new_tempo / orig_tempo

# Manual stretch entry
amt_to_stretch = 1.01587301587

print(amt_to_stretch)

y3, sr3 = sf.read(in_filename)
output_file = out_filename
# Play back at double speed
y_stretch = pyrb.time_stretch(y3, sr3, amt_to_stretch)
# Play back two semi-tones higher
y_shift = pyrb.pitch_shift(y, sr, -2)
sf.write(output_file, y_stretch, sr3, format='MP3')

y2, sr2 = librosa.load(out_filename)
tempo2, beat_frames2 = librosa.beat.beat_track(y=y2, sr=sr2)
tempo_verify = tempo2[0]
print('Estimated new tempo (for verification): {:.2f} beats per minute'.format(tempo_verify))
