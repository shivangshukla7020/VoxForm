
import speech_recognition as sr  
import pyttsx3   
import os  
from preprocessing import create_model, record_to_file, extract_feature
import streamlit as st
import time
import sys


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
            st.error("Unknown error occurred")  
            return ""  

def check_gender(male,female,if_male):
  if(if_male and female > male):
    SpeakText("Sorry you are not the correct fit for the form !")
    sys.exit()


def takeInput():
    male, female = record_audio()
    check_gender(male,female,True)
    text = recognize_speech_from_wav("test.wav")
    return text
    

# Streamlit UI  

st.title("User Input Form")  
st.subheader("Please provide your details:") 
st.caption("Welcome to our automated audio-based form! This system allows you to provide your details quickly and easily without the need for typing. You will be prompted to give information one step at a time, with 5 seconds allocated for each response. Let’s get started!.")
st.caption("Say 'START' to begin filling the form.")
 

# Input fields for user details  

first_name = st.session_state.text_input = ""
#first_name_input = st.text_input("First Name")
last_name = st.text_input("Last Name")
age = st.number_input("Age", min_value=0, max_value=120)  
roll_number = st.text_input("Roll Number")
phone_number = st.text_input("Phone Number")
city = st.text_input("City")
country = st.text_input("Country")  


# Starting Message
SpeakText("Welcome to our automated audio-based form! This system allows you to provide your details quickly and easily without the need for typing. You will be prompted to give information one step at a time, with 5 seconds allocated for each response. Let’s get started!. Say 'START' to begin filling the form.")

# Condition to start filling the form 

is_start = takeInput()
while(is_start != "start"):
  is_start = takeInput()

SpeakText("What is your first name ")
st.write(takeInput(),first_name)
#st.write(takeInput(),first_name_input)

SpeakText("What is your last name")
st.write(takeInput(),last_name)


SpeakText("Please tell your age")
st.write(takeInput(),age)


SpeakText("What is your roll number")
st.write(takeInput(),roll_number)


SpeakText("Speak up your phone number")
st.write(takeInput(),phone_number)


SpeakText("Name your city")
st.write(takeInput(),city)


SpeakText("What is your country name")
st.write(takeInput(),country)