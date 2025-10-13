import soundfile as sf
import pyrubberband as pyrb
import os

def change_tempo(folder_to_process, old_tempo):
    #change tempo to predefined tempo (128 maybe?)
    NEW_TEMPO = 128

    in_filenames = [
        f'{folder_to_process}/vocals.mp3',
        f'{folder_to_process}/drums.mp3',
        f'{folder_to_process}/other.mp3',
        f'{folder_to_process}/bass.mp3'
        ]

    # New tempo
    amt_to_stretch = NEW_TEMPO / old_tempo

    for in_filename in in_filenames:
        out_name = in_filename.split('/')[-1]
        out_filename = f'{os.getcwd()}/beat_key_matched/{folder_to_process.split('/')[-1]}/{out_name}'
        out_dir = os.path.dirname(out_filename)
        os.makedirs(out_dir, exist_ok=True)

        y3, sr3 = sf.read(in_filename)  
        output_file = out_filename
        # Play back at double speed
        y_stretch = pyrb.time_stretch(y3, sr3, amt_to_stretch)
        # Play back two semi-tones higher
        # y_shift = pyrb.pitch_shift(y3, sr3, -2)
        sf.write(output_file, y_stretch, sr3, format='MP3')

        # C:\Users\jappa\Repos\senior_project\beat_key_matched

    return f'{os.getcwd()}/beat_key_matched/{folder_to_process.split('/')[-1]}', amt_to_stretch