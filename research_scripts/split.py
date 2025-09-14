import subprocess
import os

# Define the input audio file and output directory
input_audio = 'apollo.mp3'
output_dir = 'stems_demucs'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

try:
    # Use the subprocess module to call the demucs command-line tool
    # The -n htdemucs flag specifies the model
    # The -o flag specifies the output directory
    # The -d cpu flag forces CPU usage
    command = ['python', '-m', 'demucs.separate', '-n', 'htdemucs', '-o', output_dir, '-d', 'cpu', input_audio]
    
    # Run the command and print the output
    subprocess.run(command, check=True)
    
    print(f"Stem separation complete. Stems are in the '{output_dir}/htdemucs/{os.path.splitext(os.path.basename(input_audio))[0]}' directory.")

except FileNotFoundError:
    print("Error: The 'demucs' command was not found. Please ensure Demucs is installed and in your PATH.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while running Demucs: {e.stderr}")