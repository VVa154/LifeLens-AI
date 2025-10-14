import os
import sqlite3
import datetime
import streamlit as st
import speech_recognition as sr
import requests
import random
import chromadb
from sentence_transformers import SentenceTransformer
import tempfile
from bark import SAMPLE_RATE, generate_audio
import scipy

# CONFIG
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"
DB_PATH = "user_memory.db"
THERAPIST_COLLECTION = "therapist-knowledgebase"
USER_COLLECTION = "user-conversations"
STOPWORDS_FILE ="/Users/vaishu/Documents/Applied_AI_Virtual_Env1/datasets/ai_life_coach_stopwords_expanded.txt"


# Initialize ChromaDB and Embedder
chroma_client = chromadb.PersistentClient(path="./chroma_storage")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
therapist_collection = chroma_client.get_or_create_collection(name=THERAPIST_COLLECTION)
user_collection = chroma_client.get_or_create_collection(name=USER_COLLECTION)

# Setup Database
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS memories (user TEXT, datetime TEXT, text TEXT)''')
conn.commit()

# Load Stopwords
with open(STOPWORDS_FILE, 'r') as f:
    STOPWORDS = [line.strip().lower() for line in f.readlines()]

# Functions
def save_conversation(user, user_input, ai_response):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    folder = f"conversations/{user}"
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, "chat_log.txt")
    with open(file_path, "a") as f:
        f.write(f"[{timestamp}] You: {user_input}\n")
        f.write(f"[{timestamp}] Coach: {ai_response}\n\n")
    c.execute("INSERT INTO memories VALUES (?, ?, ?)", (user, timestamp, user_input))
    conn.commit()
    user_collection.add(documents=[user_input], metadatas=[{"user": user}], ids=[user + "_" + timestamp])

def retrieve_context(user, query_text):
    therapist_results = therapist_collection.query(query_texts=[query_text], n_results=5)
    therapist_context = therapist_results.get("documents", [[]])[0]
    user_results = user_collection.query(query_texts=[query_text], n_results=5, where={"user": user})
    user_context = user_results.get("documents", [[]])[0]
    full_context = "\n".join(therapist_context + user_context)
    return full_context

def generate_response(prompt, context):
    combined_prompt = f"""
You are an AI Therapist. Use the retrieved information below to respond empathetically and helpfully.

Retrieved Information:
{context}

User Message:
{prompt}

Respond naturally like a professional therapist.
"""
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": combined_prompt,
        "stream": False
    })
    return response.json().get("response", "I'm here to listen. Could you tell me more?").strip()

def transcribe_audio(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand your audio."

def speak_response_bark(text_response):
    audio_array = generate_audio(text_response, history_prompt="v2/en_speaker_1")
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    scipy.io.wavfile.write(temp_audio_file.name, SAMPLE_RATE, audio_array)
    st.audio(temp_audio_file.name, format="audio/wav")

def clear_user_memory(user_name):
    folder = f"conversations/{user_name}"
    if os.path.exists(folder):
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))
        os.rmdir(folder)
    user_collection.delete(where={"user": user_name})

# Streamlit App
st.set_page_config(page_title="AI Life Coach", layout="wide")
st.title("AI Life Coach with Long-Term Memory")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_name = st.text_input("Enter your name to start:", key="user_name_input")

if user_name:
    st.success(f"Welcome, {user_name}! Start chatting below:")

    if st.button("Forget Me (Clear My Memory)"):
        clear_user_memory(user_name)
        st.session_state.messages = []
        st.success("Your past memory has been cleared.")

    for chat in st.session_state.messages:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    input_method = st.radio("Choose your input method:", ("Type", "Upload Audio"))
    user_input = None

    if input_method == "Type":
        user_input = st.chat_input("Say something...")

    elif input_method == "Upload Audio":
        uploaded_audio = st.file_uploader("Upload an audio file:", type=["wav", "mp3", "m4a"])
        if uploaded_audio:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_audio.read())
                user_input = transcribe_audio(tmp_file.name)
                os.unlink(tmp_file.name)
            st.info(f"You said: {user_input}")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.messages.append({"role": "user", "content": user_input})

        if any(stopword in user_input.lower() for stopword in STOPWORDS):
            ai_response = ("I'm deeply sorry you're feeling this way. Unfortunately, "
                           "I am not equipped to handle critical situations. "
                           "Please immediately contact our Head Coach at +1 1234567890 for assistance.")
            with st.chat_message("assistant"):
                st.markdown(ai_response)
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
        else:
            context = retrieve_context(user_name, user_input)
            ai_response = generate_response(user_input, context)

            with st.chat_message("assistant"):
                st.markdown(ai_response)

            st.session_state.messages.append({"role": "assistant", "content": ai_response})

            speak_response_bark(ai_response)

            save_conversation(user_name, user_input, ai_response)

conn.close()
