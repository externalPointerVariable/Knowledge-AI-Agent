from pydub import AudioSegment
# import simpleaudio as sa
from transformers import pipeline

# Load a TTS pipeline
tts = pipeline("text-to-speech", model="facebook/fastspeech2-en-ljspeech")
print("Model is initalized...")

# Convert text to speech
# text = "Hello, this is a text-to-speech conversion example."
# speech = tts(text)

# # Save the audio to a file
# with open("output_speech.wav", "wb") as f:
#     f.write(speech["audio"])


# # Load and play the generated audio
# audio = AudioSegment.from_wav("output_speech.wav")
# # play_obj = sa.play_buffer(audio.raw_data, num_channels=audio.channels, bytes_per_sample=audio.sample_width, sample_rate=audio.frame_rate)
# # play_obj.wait_done()  # Wait until playback is finished
