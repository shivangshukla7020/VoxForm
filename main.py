
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
            return -1 

def check_gender(male,female,if_male):
  if(if_male and female > male):
    SpeakText("Sorry you are not the correct fit for the form !")
    st.subheader("This form is for 'MALE' candidates only. Thank you")
    sys.exit()


def takeInput():
    male, female = record_audio()
    check = recognize_speech_from_wav('test.wav')
    if(check == -1 ):
        SpeakText("Sorry could not hear you")
        return "No input detected"
    check_gender(male,female,True)
    return check


# Streamlit UI  

st.set_page_config(page_title="Audio-Based Form", page_icon="🗣️")

placeholder = st.empty()

with placeholder.container():
    st.title("VoxForm")  
    st.markdown(
        """ 
        ### 🚀 Welcome to our automated audio-based form!  
        ## Instructions
        🗣️ This system allows you to provide your details quickly and easily without typing.  
        - You will be prompted step by step.  
        - Provide a response **one second after the question has been asked**.  
        - Say **'START'** to begin (under 30 sec).  
        ---  
        """ 
    )  
    st.write("👋 Let's get started!") 

# SpeakText("Welcome to our automated audio-based form! This system allows you to provide your details quickly and easily without typing. Please take a moment to carefully read the instructions below to ensure a smooth and seamless experience.")

time.sleep(3)
SpeakText("Say 'START' to begin filling the form.")

# Time limit to start the form
limit = time.time()

# Condition to start filling the form 
while(takeInput() != "start"):
    if time.time()-limit > 30:
        SpeakText("The program has been terminated.")
        st.write("Time Limit Exceeded")
        sys.exit()

placeholder.empty()

SpeakText("What is your first name ")
first_name = takeInput()
st.text_input(label="First Name ?",value=first_name)


SpeakText("What is your last name")
last_name = takeInput()
st.text_input("Last Name ?",value=last_name)


SpeakText("Please tell your age")
try:
    age = int(takeInput())
    st.number_input(label="Age ?",value=age)  
except ValueError:
    st.number_input(label="Age ?",value=0)  
    pass


SpeakText("What is your roll number")
roll_number = st.text_input("Roll No",value=takeInput())


SpeakText("Enter your phone number")
phone_number = st.text_input("Phone no",value=takeInput())


SpeakText("Name your city")
city = takeInput()
city = city[0].upper() + city[1:]
st.text_input("City",value=city)

SpeakText("What is your country name")
country = takeInput()
country = country[0].upper() + country[1:]
st.text_input("Country",value=country)  


st.subheader("Thank You")
SpeakText("Thank you for filling this form")