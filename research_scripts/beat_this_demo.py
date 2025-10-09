from beat_this.inference import File2Beats

file2beats = File2Beats(checkpoint_path="final0", device="cuda", dbn=False)
songs= [
    "C:\\Users\\jappa\\Repos\\senior_project\\harmonixset\\src\\mp3s\\0017_badromance.mp3",
    "C:\\Users\\jappa\\Repos\\senior_project\\harmonixset\\src\\mp3s\\0018_bassdownlow.mp3",
    "C:\\Users\\jappa\\Repos\\senior_project\\harmonixset\\src\\mp3s\\0020_becauseofyou.mp3",
    "C:\\Users\\jappa\\Repos\\senior_project\\harmonixset\\src\\mp3s\\0021_better.mp3",
    "C:\\Users\\jappa\\Repos\\senior_project\\harmonixset\\src\\mp3s\\0022_betteroffalone.mp3"
]
for audio_path in songs:
    # audio_path = "C:\\Users\\jappa\\Repos\\senior_project\\harmonixset\\src\\mp3s\\0015_babygotback.mp3"
    beats, downbeats = file2beats(audio_path)

    tempo_map = {}
    for idx in range(len(downbeats) - 1):
        calculated_tempo = round(4 * (60 / (downbeats[idx + 1] - downbeats[idx])))
        if calculated_tempo not in tempo_map:
            tempo_map[calculated_tempo] = 1
        else:
            tempo_map[calculated_tempo] += 1

    max_count = 0
    tempo = 0
    for key, val in tempo_map.items():
        if val > max_count:
            max_count = val
            tempo = key

    print(tempo)