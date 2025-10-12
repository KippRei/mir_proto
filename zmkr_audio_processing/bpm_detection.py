from beat_this.inference import File2Beats

def get_bpm(audio_path) -> int:
    file2beats = File2Beats(checkpoint_path="final0", device="cuda", dbn=False)
    beats, downbeats = file2beats(audio_path)

    tempo_map = {}
    start_beat_at_tempo_map = {}
    for idx in range(len(downbeats) - 1):
        calculated_tempo = round(4 * (60 / (downbeats[idx + 1] - downbeats[idx])), 4)
        print(f'tempo: {calculated_tempo}, downbeat: {downbeats[idx]}')
        if calculated_tempo not in tempo_map:
            tempo_map[calculated_tempo] = 1
            # TODO: Check for possible out of range
            start_beat_at_tempo_map[calculated_tempo] = downbeats[idx]
        else:
            tempo_map[calculated_tempo] += 1

    max_count = 0
    tempo = 0
    for key, val in tempo_map.items():
        if val > max_count:
            max_count = val
            tempo = key

    print(f'{tempo}: start beat= {start_beat_at_tempo_map[tempo]}')  
    return tempo, start_beat_at_tempo_map[tempo]