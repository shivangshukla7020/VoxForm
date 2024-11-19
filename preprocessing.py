import wave
import librosa
import numpy as np
from sys import byteorder
from array import array
from struct import pack
from audio_recorder_streamlit import audio_recorder
import streamlit as st  

# Constants for audio processing  
RATE = 16000  
THRESHOLD = 500  # Silence threshold  
MAXIMUM = 16384  
SILENCE_DURATION_LIMIT = 3  # Stop recording after 3 seconds of silence  


class AudioProcessor:  
    """Custom audio processor for detecting silence and storing audio data."""  
    def __init__(self):  
        self.frames = []  
        self.silence_duration = 0  # Counter for silence duration  
        self.is_speaking = True  

    def process_audio(self, audio_data):  
        """Process incoming audio data."""  
        if is_silent(audio_data):  
            self.silence_duration += 1  
        else:  
            self.silence_duration = 0  
            self.frames.append(audio_data)  

        # Stop recording after prolonged silence  
        if self.silence_duration > SILENCE_DURATION_LIMIT * RATE / 1024:  # Assuming frame size of 1024  
            self.is_speaking = False  

    def get_audio_data(self):  
        """Retrieve the recorded audio data."""  
        if self.frames:  
            return np.concatenate(self.frames)  
        return np.array([])  

def is_silent(audio_data):  
    """Check if the audio data is silent."""  
    return max(audio_data) < THRESHOLD  

def trim(audio_data):  
    """Trim the silent parts from the start and end of the audio data."""  
    return np.trim_zeros(audio_data, trim="fb")  

def normalize(audio_data):  
    """Normalize the audio to a consistent volume."""  
    times = float(MAXIMUM) / max(abs(i) for i in audio_data)  
    return (audio_data * times).astype(np.int16)  

def record_to_file(output_file="output.wav"):  
    """  
    Automatically records audio until silence is detected and saves it to a WAV file.  

    Parameters:  
    - output_file: Path to save the WAV file.  
    """  
    # Create an audio processor  
      

    # Process audio data if available  
    if audio_data is not None:  
        audio_processor.process_audio(audio_data)  

    # Retrieve and process audio data  
    audio_data = audio_processor.get_audio_data()  
    if len(audio_data) == 0:  
        st.warning("No audio recorded.")  
        return  

    # Trim and normalize audio  
    audio_data = normalize(trim(audio_data))  
    audio_data = array("h", audio_data)  
    audio_bytes = pack("<" + ("h" * len(audio_data)), *audio_data)  

    # Save audio to a WAV file  
    with wave.open(output_file, "wb") as wf:  
        wf.setnchannels(1)  # Mono channel  
        wf.setsampwidth(2)  # 16-bit PCM  
        wf.setframerate(RATE)  
        wf.writeframes(audio_bytes)  

    st.success(f"Audio saved to {output_file}")



def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    X, sample_rate = librosa.core.load(file_name)
    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        result = np.hstack((result, tonnetz))
    return result



def create_model(vector_length=128):
    import tensorflow as tf
  
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(256, input_shape=(vector_length,)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    # one output neuron with sigmoid activation function, 0 means female, 1 means male
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    # using binary crossentropy as it's male/female classification (binary)
    model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")
    # print summary of the model
    model.summary()
    return model



