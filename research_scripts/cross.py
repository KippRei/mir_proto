import librosa
import matplotlib.pyplot as plt

hop_length = 1024
y_ref, sr = librosa.load('twofriends.mp3')
y_comp, sr = librosa.load('twofriends.mp3')
chroma_ref = librosa.feature.chroma_cqt(y=y_ref, sr=sr, hop_length=hop_length)
chroma_comp = librosa.feature.chroma_cqt(y=y_comp, sr=sr, hop_length=hop_length)
# Use time-delay embedding to get a cleaner recurrence matrix
x_ref = librosa.feature.stack_memory(chroma_ref, n_steps=10, delay=3)
x_comp = librosa.feature.stack_memory(chroma_comp, n_steps=10, delay=3)
xsim = librosa.segment.cross_similarity(x_comp, x_ref)
xsim_aff = librosa.segment.cross_similarity(x_comp, x_ref, metric='cosine', mode='affinity')

for i in xsim:
    print(i)

fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
imgsim = librosa.display.specshow(xsim, x_axis='s', y_axis='s',
                         hop_length=hop_length, ax=ax[0])
ax[0].set(title='Binary cross-similarity (symmetric)')
imgaff = librosa.display.specshow(xsim_aff, x_axis='s', y_axis='s',
                         cmap='magma_r', hop_length=hop_length, ax=ax[1])
ax[1].set(title='Cross-affinity')
ax[1].label_outer()
fig.colorbar(imgsim, ax=ax[0], orientation='horizontal', ticks=[0, 1])
fig.colorbar(imgaff, ax=ax[1], orientation='horizontal')
plt.show()