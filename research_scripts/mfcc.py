import librosa

y, sr = librosa.load("twofriends.mp3")

mfcc_arr = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=20, n_fft=4096, n_mels=256)

for idx, m in enumerate(mfcc_arr):
    print(str(idx) + ": " + str(m))