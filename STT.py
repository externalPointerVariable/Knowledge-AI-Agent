from transformers import pipeline

# Load Whisper model (large model is recommended for high accuracy)
speech_to_text = pipeline(model="openai/whisper-large", device="cpu")  # or "cuda" for GPU
print("Model is initialized...")

# Transcribe audio
transcription = speech_to_text("path/to/your_audio.wav")
print(transcription['text'])
