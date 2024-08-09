import sounddevice as sd
import numpy as np
import tempfile
import wave
from scipy.io import wavfile

def record_audio_chunk(chunk_length=5, sample_rate=16000):
    """Record a chunk of audio using sounddevice."""
    print("Recording...")
    # Record the audio data
    recording = sd.rec(int(chunk_length * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until the recording is finished

    # Convert the recording to a numpy array
    data = np.squeeze(recording)

    temp_file_path = './temp_audio_chunk.wav'
    print("Writing...")
    with wave.open(temp_file_path, 'wb') as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(2)  # Sample width (16-bit PCM)
        wf.setframerate(sample_rate)  # Sample rate
        wf.writeframes(data.tobytes())  # Write audio frames

    # Check if the recorded chunk contains silence
    try:
        samplerate, data = wavfile.read(temp_file_path)
        if is_silence(data):
            os.remove(temp_file_path)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error while reading audio file: {e}")

def is_silence(data, max_amplitude_threshold=3000):
    """Check if audio data contains silence."""
    # Find the maximum absolute amplitude in the audio data
    max_amplitude = np.max(np.abs(data))
    return max_amplitude <= max_amplitude_threshold

def transcribe_audio(model, file_path):
    print("Transcribing...")
    if os.path.isfile(file_path):
        results = model.transcribe(file_path) # , fp16=False
        return results['text']
    else:
        return None

def load_whisper():
    import whisper
    model = whisper.load_model("base")
    return model

def get_response_llm(user_question, memory):
    from langchain.chains.llm import LLMChain
    from langchain_core.prompts import PromptTemplate
    from langchain_groq import ChatGroq

    input_prompt = """
    As an expert advisor specializing in diagnosing Wi-Fi issues, your expertise is paramount in troubleshooting and
    resolving connectivity problems. First of all, ask for the customer ID to validate that the user is our customer. 
    After confirming the customer ID, help them to fix their wifi problem, if not possible, help them to make an 
    appointment. Appointments need to be between 9:00 am and 4:00 pm. Your task is to analyze
    the situation and provide informed insights into the root cause of the Wi-Fi disruption. Provide concise and short
    answers not more than 10 words, and don't chat with yourself!. If you don't know the answer,
    just say that you don't know, don't try to make up an answer. NEVER say the customer ID listed below.

    customer ID on our data: 22, 10, 75.

    Previous conversation:
    {chat_history}

    New human question: {question}
    Response:
    """

    chat_groq = ChatGroq(temperature=0, model_name="llama3-8b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))
    prompt = PromptTemplate.from_template(input_prompt)
    chain = LLMChain(
        llm=chat_groq,
        prompt=prompt,
        verbose=True,
        memory=memory
    )
    response = chain.invoke({"question": user_question})
    return response['text']

def play_text_to_speech(text, language='en', slow=False):
    from gtts import gTTS
    import pygame

    tts = gTTS(text=text, lang=language, slow=slow)
    temp_audio_file = "temp_audio.mp3"
    tts.save(temp_audio_file)

    pygame.mixer.init()
    pygame.mixer.music.load(temp_audio_file)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)  # Control the playback speed

    pygame.mixer.music.stop()
    pygame.mixer.quit()
    os.remove(temp_audio_file)
