from pydub import AudioSegment

# Load your audio files with pydub
song1 = AudioSegment.from_file("stems_demucs/htdemucs/apollo/bass.wav")
song2 = AudioSegment.from_file("stems_demucs/htdemucs/apollo/drums.wav")
song3 = AudioSegment.from_file("stems_demucs/htdemucs/apollo/other.wav")

song4 = AudioSegment.from_file("stretched_pitched_friends_vocal.mp3")


# You can also manually overlay them for more control
# This is useful for creating a mashup of two songs playing simultaneously
# You would first align the songs based on BPM/beats, then use overlay.
overlay_start_time = 37000  # Overlay song2 10 seconds into song1
overlayed_mix = song1.overlay(song2.overlay(song3))
final = overlayed_mix.overlay(song4, overlay_start_time)

# Export the final mix
overlayed_mix.export("mixed_song.mp3", format="mp3")