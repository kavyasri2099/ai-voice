import os
import sounddevice as sd
import streamlit as st
from langchain.memory import ConversationBufferMemory
from scipy.io import wavfile
import numpy as np

from utils import record_audio_chunk, transcribe_audio, get_response_llm, play_text_to_speech, load_whisper

chunk_file = 'temp_audio_chunk.wav'
model = load_whisper()

def record_audio_chunk(duration=5):
    """Record audio for a specified duration and save to a file."""
    print("Recording...")
    sample_rate = 16000
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished

    # Convert to numpy array
    audio_data = audio_data.flatten()
    
    # Check if the recorded chunk contains silence
    if is_silence(audio_data):
        return None
    
    # Save audio data to a temporary file
    wavfile.write(chunk_file, sample_rate, audio_data)
    return chunk_file

def is_silence(data, max_amplitude_threshold=3000):
    """Check if audio data contains silence."""
    max_amplitude = np.max(np.abs(data))
    return max_amplitude <= max_amplitude_threshold

def main():
    st.markdown('<h1 style="color: darkblue;">AI Voice Assistant️</h1>', unsafe_allow_html=True)

    memory = ConversationBufferMemory(memory_key="chat_history")

    if st.button("Start Recording"):
        while True:
            # Record and save audio chunk
            record_audio_chunk()

            text = transcribe_audio(model, chunk_file)

            if text is not None:
                st.markdown(
                    f'<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">Customer 👤: {text}</div>',
                    unsafe_allow_html=True)

                os.remove(chunk_file)

                response_llm = get_response_llm(user_question=text, memory=memory)
                st.markdown(
                    f'<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">AI Assistant 🤖: {response_llm}</div>',
                    unsafe_allow_html=True)

                play_text_to_speech(text=response_llm)
            else:
                break  # Exit the while loop

        print("End Conversation")

if __name__ == "__main__":
    main()
