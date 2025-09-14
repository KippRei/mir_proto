import whisper
import os

def transcribe_audio_with_whisper(audio_file_path, model_size="base"):
    """
    Transcribes an audio file using the OpenAI Whisper model.

    Args:
        audio_file_path (str): The path to the audio file to transcribe.
        model_size (str): The size of the Whisper model to use.
                          Options: "tiny", "base", "small", "medium", "large".
                          Larger models are more accurate but slower.
    
    Returns:
        str: The transcribed text.
    """
    if not os.path.exists(audio_file_path):
        print(f"Error: The audio file '{audio_file_path}' was not found.")
        return None

    try:
        # Load the Whisper model.
        # This will download the model the first time you run it.
        print(f"Loading Whisper model '{model_size}'...")
        model = whisper.load_model(model_size)

        # Transcribe the audio.
        # This is where the magic happens.
        print("Transcribing audio... This may take a moment.")
        result = model.transcribe(audio_file_path)

        # The result object is a dictionary. The 'text' key contains the transcription.
        transcribed_text = result["text"]
        print("Transcription successful.")
        
        return transcribed_text

    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        return None

# --- Main part of the script ---
if __name__ == "__main__":
    
    # You need to provide a path to an audio file (e.g., .mp3, .wav).
    # Replace 'path/to/your/audio.mp3' with a real file path.
    # For a song, this would be your 'vocals.mp3' stem.
    audio_file_to_transcribe = 'twofriends.mp3'
    
    # # --- Example 1: Use a small model for a quick result ---
    # print("--- Running transcription with 'small' model ---")
    # transcribed_text = transcribe_audio_with_whisper(audio_file_to_transcribe, model_size="small")
    
    # if transcribed_text:
    #     print("\n--- Transcribed Text ---")
    #     print(transcribed_text)
    
    # print("\n" + "="*50 + "\n")

    # --- Example 2: Use a more accurate but slower model ---
    # This is a good option if you have a decent GPU.
    # You can comment this out if you're just testing on a CPU.
    print("--- Running transcription with 'medium' model ---")
    transcribed_text_medium = transcribe_audio_with_whisper(audio_file_to_transcribe, model_size="medium")

    if transcribed_text_medium:
        print("\n--- Transcribed Text (medium model) ---")
        print(transcribed_text_medium)