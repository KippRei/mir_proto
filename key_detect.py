import librosa
import numpy as np

# 1. Load an audio file
y, sr = librosa.load(librosa.ex('nutcracker'), duration=30)

# 2. Compute the chroma energy normalized statistics (CENS) features.
chroma = librosa.feature.chroma_cens(y=y, sr=sr)

# 3. Compute a global chroma profile by taking the median of the chromagram.
chroma_profile = np.median(chroma, axis=1)

# 4. Define the 24 tonal key templates manually.
# These are the standard Krumhansl-Schmuckler key profiles.
major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])

# We create 12 major templates and 12 minor templates by rotating the base profile.
key_templates = []
for i in range(12):
    # Add major key templates
    key_templates.append(np.roll(major_template, i))
for i in range(12):
    # Add minor key templates
    key_templates.append(np.roll(minor_template, i))

key_templates = np.array(key_templates)

# 5. Compare the chroma profile to the templates to find the best match.
# The best key is the one with the highest correlation (dot product).
best_key_index = np.argmax(np.dot(key_templates, chroma_profile))

# 6. Map the index back to a key name.
key_names = [
    'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
    'c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b'
]
estimated_key = key_names[best_key_index]

print(f"The estimated key of the song is: {estimated_key}")