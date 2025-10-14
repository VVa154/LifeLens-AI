import os
import sqlite3
import datetime
import streamlit as st
import speech_recognition as sr
import requests
import random
import chromadb
import json
from sentence_transformers import SentenceTransformer
import pyttsx3
import tempfile

# CONFIG
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"
DB_PATH = "user_memory.db"
THERAPIST_COLLECTION = "therapist-knowledgebase"
USER_COLLECTION = "user-conversations"
STOPWORDS_FILE = "/Users/vaishu/Documents/Applied_AI/datasets/ai_life_coach_stopwords_expanded.txt"
FINAL_DATASET_FILE = "/Users/vaishu/Documents/Applied_AI/datasets/final_combined_dataset.json"

# Initialize ChromaDB and Embedder
chroma_client = chromadb.PersistentClient(path="./chroma_storage")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
therapist_collection = chroma_client.get_or_create_collection(name=THERAPIST_COLLECTION)
user_collection = chroma_client.get_or_create_collection(name=USER_COLLECTION)

# Embed final_combined_dataset.json if therapist_collection is empty
if len(therapist_collection.get()["ids"]) == 0:
    with open(FINAL_DATASET_FILE, "r") as f:
        data = json.load(f)
    for idx, item in enumerate(data):
        question = item.get("question", "")
        answer = item.get("answer", "")
        if question and answer:
            combined_text = f"Q: {question}\nA: {answer}"
            therapist_collection.add(
                documents=[combined_text],
                metadatas=[{"source": "therapist_dataset"}],
                ids=[f"therapist_{idx}"]
            )

# Setup Database
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS memories (user TEXT, datetime TEXT, text TEXT)''')
conn.commit()

# Load Stopwords
with open(STOPWORDS_FILE, 'r') as f:
    STOPWORDS = [line.strip().lower() for line in f.readlines()]

# Save conversation
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

# Get user memory summary
def retrieve_context(user):
    c.execute("SELECT datetime, text FROM memories WHERE user=? ORDER BY datetime ASC", (user,))
    rows = c.fetchall()

    if not rows or len(rows) <= 1:
        return ""

    full_conversation = "\n".join([f"[{timestamp}] {text}" for timestamp, text in rows])

    summary_prompt = f"""
You are a helpful AI assistant summarizer.

Here is the full conversation history:
{full_conversation}

Summarize this conversation into 5-7 lines, focusing on main topics discussed, emotions shared, and overall progress made.
Be concise but empathetic.
"""

    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": summary_prompt, "stream": False}
    )

    return response.json().get("response", "").strip()

# Get therapist knowledge
def get_therapist_knowledge(user_input):
    results = therapist_collection.query(query_texts=[user_input], n_results=5)
    therapist_contexts = results.get("documents", [[]])[0]
    return "\n".join(therapist_contexts)

# Generate final response
def generate_response(prompt, user_context, therapist_context, user_name, first_response_done):
    # Custom greeting based on user history
    if not first_response_done:
        if user_context.strip() == "":
            greeting = f"Hey, looks like we are meeting for the first time—how have you been?\nWelcome to LifeLens AI!\n\n"
        else:
            greeting = f"Hi {user_name}, welcome to LifeLens AI.\n\n"
    else:
        greeting = ""

    # Tell the model to avoid repeating greeting
    skip_greeting_instruction = (
        "Important: You do NOT need to greet the user (e.g., 'Hi there', 'Hello'). I’ve already done that.\n\n"
        if not first_response_done else
        "Important: DO NOT use greetings like 'Hi', 'Hello', 'Hi there', or similar. Start directly with emotional reflection or helpful response.\n\n"
    )

    # Add context reference if it exists, otherwise explicitly state not to fabricate
    if user_context.strip() != "":
        context_instruction = f"""
Important Rule:
- The user has chatted with you before.
- Begin your response by naturally referencing the following summary.
- DO NOT fabricate or guess what was said — only use the summary below.

Conversation Summary:
{user_context}
"""
    else:
        context_instruction = """
Important Rule:
- This is the user's first message.
- DO NOT say anything that assumes a previous chat or session.
- Focus on responding to this message alone, naturally and empathetically.
"""

    # Final system prompt construction
    system_prompt = f"""
You are an AI Therapist and Life Coach with long-term memory, focused on emotional intelligence, human-like warmth, and natural conversation.

{skip_greeting_instruction}
Behavior Guidelines:
- Never repeat the user's name after the first message.
- Avoid generic greetings like 'Hi there', 'Hello'.
- Refer to real previous memory only if provided.
- Prioritize warmth, empathy, and helpful tone.
- Keep replies concise (3–5 lines).

Therapist Knowledge:
{therapist_context}
{context_instruction}

User's New Message:
{prompt}

Now respond naturally and directly.
"""

    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": system_prompt, "stream": False}
    )

    return greeting + response.json().get("response", "I'm here for you whenever you're ready to chat.").strip()


# Voice transcription
def transcribe_audio(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand your audio."

# Text-to-speech
def speak_response(text_response):
    engine = pyttsx3.init()
    engine.say(text_response)
    engine.runAndWait()

# Clear memory
def clear_user_memory(user_name):
    folder = f"conversations/{user_name}"
    if os.path.exists(folder):
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))
        os.rmdir(folder)
    user_collection.delete(where={"user": user_name})

# Streamlit UI
st.set_page_config(page_title="LifeLens AI", layout="wide")
st.title("LifeLens AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "first_response_done" not in st.session_state:
    st.session_state.first_response_done = False

user_name = st.text_input("Enter your name to start:", key="user_name_input")

if user_name:
    st.success(f"Welcome, {user_name}! Start chatting below:")

    if st.button("Forget Me (Clear My Memory)"):
        clear_user_memory(user_name)
        st.session_state.messages = []
        st.session_state.first_response_done = False
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

        goodbye_words = ["bye", "goodbye", "see you", "take care"]

        if any(word in user_input.lower() for word in goodbye_words):
            goodbye_response = "Please take care."
            with st.chat_message("assistant"):
                st.markdown(goodbye_response)
            st.session_state.messages.append({"role": "assistant", "content": goodbye_response})
            speak_response(goodbye_response)
            save_conversation(user_name, user_input, goodbye_response)

        else:
            user_context = retrieve_context(user_name)
            therapist_context = get_therapist_knowledge(user_input)

            if any(stopword in user_input.lower() for stopword in STOPWORDS):
                ai_response = (
                    "I'm deeply sorry you're feeling this way. Unfortunately, "
                    "I am not equipped to handle critical situations. "
                    "Please immediately contact our Head Coach at +1 1234567890 for assistance."
                )
            else:
                ai_response = generate_response(
                    user_input,
                    user_context,
                    therapist_context,
                    user_name,
                    st.session_state.first_response_done
                )

            with st.chat_message("assistant"):
                st.markdown(ai_response)
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            speak_response(ai_response)
            save_conversation(user_name, user_input, ai_response)
            st.session_state.first_response_done = True

conn.close()
