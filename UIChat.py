import os
import json
from datetime import datetime
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import speech_recognition as sr
import pyttsx3

# === Chat History Disk Persistence ===
HISTORY_DIR = "chat_history"
os.makedirs(HISTORY_DIR, exist_ok=True)

def get_history_filepath():
    session_id = "chat1"  # For now, static session ID
    return os.path.join(HISTORY_DIR, f"{session_id}.json")

def load_history():
    filepath = get_history_filepath()
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return []

def save_history(messages):
    filepath = get_history_filepath()
    with open(filepath, "w") as f:
        json.dump([{"type": msg.type, "content": msg.content} for msg in messages], f, indent=2)

# === Voice Engine Setup ===
engine = pyttsx3.init()
engine.setProperty('rate', 175)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        st.warning("Could not understand audio")
    except sr.RequestError as e:
        st.error(f"Speech recognition error: {e}")
    return ""

# === Streamlit UI Setup ===
st.set_page_config(page_title="Local Chatbot", page_icon="")
st.title("Daniel's ChatBot")

# === Sidebar System Prompt ===
with st.sidebar:
    st.subheader("🤖 Bot Personality")
    system_prompt = st.text_area("Set system prompt (personality)", value="You are a helpful, witty, concise assistant.")
    use_voice_input = st.checkbox("🎙️ Use voice input")
    use_voice_output = st.checkbox("🔊 Use voice output")

# === Initialise Session State ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()
    st.session_state.messages = []

    for msg in load_history():
        if msg["type"] == "human":
            st.session_state.chat_history.add_user_message(msg["content"])
            st.session_state.messages.append(HumanMessage(content=msg["content"]))
        elif msg["type"] == "ai":
            st.session_state.chat_history.add_ai_message(msg["content"])
            st.session_state.messages.append(AIMessage(content=msg["content"]))

# === LLM Setup ===
llm = ChatOllama(model="llama3.2:3b")
retriever = None

# === Document Upload ===
uploaded_file = st.file_uploader("\U0001F4C4 Upload a PDF or .txt file", type=["pdf", "txt"])

if uploaded_file is not None:
    with open("uploaded_doc", "wb") as f:
        f.write(uploaded_file.read())

    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader("uploaded_doc")
    else:
        loader = TextLoader("uploaded_doc")

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="llama3.2:3b")
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever()

# === Display Chat History ===
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user", avatar="\U0001F464").write(msg.content)
    else:
        st.chat_message("assistant", avatar="\U0001F916").write(msg.content)

# === Chat Input ===
user_input = None
if use_voice_input:
    if st.button("🎙️ Speak"):
        user_input = recognize_speech()
else:
    user_input = st.chat_input("Ask me anything...")

if user_input:
    st.chat_message("user", avatar="\U0001F464").write(user_input)
    st.session_state.chat_history.add_user_message(user_input)

    if retriever:
        context_docs = retriever.invoke(user_input)
        context_text = "\n\n".join([doc.page_content for doc in context_docs[:3]])
        user_input_with_context = f"Context:\n{context_text}\n\nQuestion: {user_input}"
    else:
        user_input_with_context = user_input

    if not st.session_state.messages:
        st.session_state.chat_history.add_ai_message(system_prompt)

    response = llm.invoke(st.session_state.chat_history.messages + [HumanMessage(content=user_input_with_context)])

    st.session_state.chat_history.add_ai_message(response.content)
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.session_state.messages.append(AIMessage(content=response.content))
    save_history(st.session_state.messages)

    st.chat_message("assistant", avatar="\U0001F916").write(response.content)

    if use_voice_output:
        speak(response.content)
