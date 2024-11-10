
import speech_recognition as sr  
import pyttsx3   
import os  
from preprocessing import create_model, record_to_file, extract_feature
import streamlit as st
import time
import sys

# Gender Initialisation
gender = "Male"

# load the saved model (after training)
# construct the model
model = create_model()
# load the saved/trained weights
model.load_weights("Model/model.h5")


def record_audio():
  # Record the audio
  print("Please talk")
  file = "test.wav"
  record_to_file(file)

  # extract features and reshape it
  features = extract_feature(file, mel=True).reshape(1, -1)
  # predict the gender!
  male_prob = model.predict(features)[0][0]
  female_prob = 1 - male_prob
  gender = "male" if male_prob > female_prob else "female"
  #show the result!
  print("Result:", gender)
  print(f"Probabilities:     Male: {male_prob*100:.2f}%    Female: {female_prob*100:.2f}%")

  return male_prob, female_prob

# Initialize the recognizer  
r = sr.Recognizer()  

# Function to convert text to speech  
def SpeakText(command):  
    engine = pyttsx3.init()  
    engine.say(command)   
    engine.runAndWait()  

# Function to recognize speech from a WAV file  
def recognize_speech_from_wav(file_path):  
    # Check if the file exists  
    if not os.path.exists(file_path):  
        st.error(f"The file {file_path} does not exist.")  
        return ""  # Return empty string if file doesn't exist  
    
    # Load the audio file  
    with sr.AudioFile(file_path) as source:  
        # Adjust for ambient noise  
        r.adjust_for_ambient_noise(source, duration=0.2)  
        
        # Record the audio from the file  
        audio_data = r.record(source)  

        # Using Google to recognize audio  
        try:  
            MyText = r.recognize_google(audio_data)  
            MyText = MyText.lower()  
            return MyText  # Return recognized text  
        except sr.RequestError as e:  
            st.error(f"Could not request results {0}".format(e))  
            return ""  
        except sr.UnknownValueError:   
            return "No input detected"  

def check_gender(male,female,if_male):
  if(if_male and female > male):
    SpeakText("Sorry you are not the correct fit for the form !")
    st.subheader("This form is for 'MALE' candidates only. Thank you")
    sys.exit()


def takeInput():
    if(not check_voice_activity):
        SpeakText("Sorry could not hear you")
        return "No input detected"
    male, female = record_audio()
    check_gender(male,female,True)
    text = recognize_speech_from_wav("test.wav")
    return text

import librosa  
import numpy as np  

import librosa  
import numpy as np  

def check_voice_activity(wav_file, threshold=0.1):  
    """  
    Check for presence of voice activity in a WAV file.  
    
    Parameters:  
    wav_file (str): Path to the WAV file.  
    threshold (float): Energy threshold to consider for voice activity.  
    
    Returns:  
    bool: True if voice activity is detected, False otherwise.  
    """  
    # Load the audio file  
    y, sr = librosa.load(wav_file, sr=None)  
    
    # Calculate the short-time energy of the audio signal  
    frame_length = int(sr * 0.5)  # Fixed frame length for 0.5 seconds  
    hop_length = frame_length // 2  # Overlap between frames  
    energy = np.array([  
        np.sum(np.square(y[i:i + frame_length]))  
        for i in range(0, len(y), hop_length)  
        if i + frame_length <= len(y)  
    ])  
    
    # Check if the energy exceeds the threshold  
    voice_activity_detected = np.any(energy > threshold)  

    return voice_activity_detected

    

# Streamlit UI  

st.title(f"User Input Form (Target = {gender})")  
st.subheader("Please provide your details:") 
st.caption("Welcome to our automated audio-based form! This system allows you to provide your details quickly and easily without the need for typing. You will be prompted to give information one step at a time, with 5 seconds allocated for each response. Let’s get started!.")

  

# Starting Message
SpeakText("Welcome to our automated audio-based form! This system allows you to provide your details quickly and easily without the need for typing. You will be prompted to give information one step at a time, with 5 seconds allocated for each response. Let’s get started!.")

st.subheader("Say 'START' to begin filling the form.")
SpeakText("Say 'START' to begin filling the form.")

# Condition to start filling the form 

while(takeInput() != "start"):
    pass

# Function to add custom CSS  
def add_custom_css():  
    st.markdown(  
        """  
        <style>  
    /* Increase specificity for input fields */  
    div[data-testid="stTextInput"] > div > input,   
    div[data-testid="stNumberInput"] > div > input {  
        background-color: #e0f7fa;  
        border: 2px solid #009688;  
        border-radius: 5px;  
        padding: 10px;  
        font-size: 18px;  
        color: #333;  
        opacity : 1 !important;
        transition: opacity 0s !important;  
    }  

    /* Ensure full opacity for inputs and labels */  
    div[data-testid="stTextInput"] > div > input:disabled,   
    div[data-testid="stNumberInput"] > div > input:disabled {  
        background-color: #e0f7fa;  
        border: 2px solid #00796b;  
        color: #333;  
        opacity: 1 !important;  
    }  

    div[data-testid="stTextInput"] > div > label,   
    div[data-testid="stNumberInput"] > div > label {  
        color: #00796b;  
        font-weight: bold;  
        opacity: 1 !important;  
    }  
    </style>    
        """,  
        unsafe_allow_html=True  
    )  

# Adding custom CSS  
add_custom_css() 


SpeakText("What is your first name ")
first_name = takeInput()
st.text_input(label="First Name ?",value=first_name,disabled=True)


SpeakText("What is your last name")
last_name = takeInput()
st.text_input("Last Name ?",value=last_name,disabled=True)


SpeakText("Please tell your age")
try:
    age = int(takeInput())
    st.number_input(label="Age ?",value=age,disabled=True)  
except ValueError:
    st.number_input(label="Age ?",value=0,disabled=True)  
    pass


SpeakText("What is your roll number")
roll_number = st.text_input("Roll No",value=takeInput(),disabled=True)


SpeakText("Enter your phone number")
phone_number = st.text_input("Phone no",value=takeInput(),disabled=True)


SpeakText("Name your city")
city = takeInput()
city = city[0].upper() + city[1:]
st.text_input("City",value=city,disabled=True)

SpeakText("What is your country name")
country = takeInput()
country = country[0].upper + country[1:]
st.text_input("Country",value=country,disabled=True)  


st.subheader("Thank You")
SpeakText("Thank you for filling this form")