import os
import streamlit as st
from langchain.memory import ConversationBufferMemory
from utils import record_audio_chunk, transcribe_audio, get_response_llm, play_text_to_speech, load_whisper

chunk_file = 'temp_audio_chunk.wav'
model = load_whisper()

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
