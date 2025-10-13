from beat_this.inference import File2Beats

def get_bpm(audio_path) -> int:
    file2beats = File2Beats(checkpoint_path="final0", device="cuda", dbn=False)
    beats, downbeats = file2beats(audio_path)

    # Get tempo via downbeats
    tempo_map = {}
    start_beat_at_tempo_map = {}
    for idx in range(len(downbeats) - 1):
        calculated_tempo = round(4 * (60 / (downbeats[idx + 1] - downbeats[idx])), 10)
        # print(f'tempo: {calculated_tempo}, downbeat: {downbeats[idx]}')
        if calculated_tempo not in tempo_map:
            tempo_map[calculated_tempo] = 1
            start_beat_at_tempo_map[calculated_tempo] = downbeats[idx]
        else:
            tempo_map[calculated_tempo] += 1

    sorted_tempo_keys = sorted(list(tempo_map.keys()))
    max_count = 0
    tempo = 0
    best_tempo = -1
    prev_bpm = -1
    next_bpm = -1

    for idx, key in enumerate(sorted_tempo_keys):
        if tempo_map[key] > max_count:
            max_count = tempo_map[key]
            best_tempo = key
            if idx == 0:
                prev_bpm = -1
                next_bpm = sorted_tempo_keys[idx + 1]
            elif idx == len(sorted_tempo_keys) - 1:
                prev_bpm = sorted_tempo_keys[idx - 1]
                next_bpm = -1
            else:
                prev_bpm = sorted_tempo_keys[idx - 1]
                next_bpm = sorted_tempo_keys[idx + 1]               

    prev_bpm_count = tempo_map[prev_bpm] if prev_bpm != -1 else 0
    next_bpm_count = tempo_map[next_bpm] if next_bpm != -1 else 0

    # Average values using MSE inspired calculation
    total_bpm_samples = pow(tempo_map[best_tempo], 2) + pow(prev_bpm_count, 2) + pow(next_bpm_count, 2)
    summed_tempos = (best_tempo * pow(tempo_map[best_tempo], 2)) + (prev_bpm * pow(prev_bpm_count, 2)) + (next_bpm * pow(next_bpm_count, 2))
    average_tempo = summed_tempos / total_bpm_samples
    for key, val in tempo_map.items():
        print(f'BPM: {key}, Count: {val}')
        if val > max_count:
            max_count = val
            tempo = key

    print(f'{average_tempo}')  
    return average_tempo, downbeats