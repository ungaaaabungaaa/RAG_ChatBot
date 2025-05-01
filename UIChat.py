import os
import json
import hashlib
import time
import streamlit as st
import shutil
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import speech_recognition as sr
import pyttsx3
import ollama
from vosk import Model, KaldiRecognizer
import wave

# === Configuration ===
DEFAULT_LLM_MODEL = "mistral:7b"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:latest"
DEFAULT_SYSTEM_PROMPT = "You are Chatbot, a helpful, witty, and slightly sarcastic assistant. Answer concisely based *only* on the provided context documents. If the answer isn't in the context, say you don't know."
HISTORY_DIR = "chat_history"
TEMP_DIR = "temp_docs"
VOSK_MODEL_PATH = "vosk-models/vosk-model-small-en-us-0.15"  # Adjust to your Vosk model path
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# === Page Configuration ===
st.set_page_config(page_title="Local RAG Chatbot with Voice", page_icon="📚", layout="wide")
st.title("📚 ChatBot (Multi-PDF) with Voice")

# === Chat History Disk Persistence ===
def get_history_filepath():
    session_id = "current_chat"
    return os.path.join(HISTORY_DIR, f"{session_id}.json")

@st.cache_data(ttl=300)
def load_history():
    """Loads chat history from a JSON file."""
    filepath = get_history_filepath()
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                history_json = json.load(f)
                history = []
                for msg_data in history_json:
                    if msg_data.get("type") == "human":
                        history.append(HumanMessage(content=msg_data["content"]))
                    elif msg_data.get("type") == "ai":
                        history.append(AIMessage(content=msg_data["content"]))
                return history
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            st.error(f"Error loading chat history: {e}. Starting fresh.")
            return []
    return []

def save_history(messages):
    """Saves chat history to a JSON file."""
    filepath = get_history_filepath()
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            serializable_messages = [
                {"type": "human" if isinstance(msg, HumanMessage) else "ai", "content": msg.content}
                for msg in messages if isinstance(msg, (HumanMessage, AIMessage))
            ]
            json.dump(serializable_messages, f, indent=2, ensure_ascii=False)
        if "save_triggered" in st.session_state:
            del st.session_state.save_triggered
    except Exception as e:
        st.error(f"Error saving chat history: {e}")

# === Ollama Model Listing ===
@st.cache_data(ttl=600)
def get_available_ollama_models():
    """Fetches the list of locally available Ollama models."""
    try:
        model_list = ollama.list()
        return [model['name'] for model in model_list.get('models', [])]
    except Exception:
        return [DEFAULT_LLM_MODEL, DEFAULT_EMBEDDING_MODEL]

# === Initialize Session State ===
if "messages" not in st.session_state:
    st.session_state.messages = load_history()

if "llm" not in st.session_state:
    try:
        st.session_state.llm = ChatOllama(model=DEFAULT_LLM_MODEL)
    except Exception as e:
        st.error(f"Failed to initialize default LLM {DEFAULT_LLM_MODEL}: {e}. Ensure Ollama is running and the model is pulled.")
        st.session_state.llm = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "processed_files_hash" not in st.session_state:
    st.session_state.processed_files_hash = None

if "processed_file_names" not in st.session_state:
    st.session_state.processed_file_names = []

if "file_uploader_key_counter" not in st.session_state:
    st.session_state.file_uploader_key_counter = 0

if "tts_engine" not in st.session_state:
    st.session_state.tts_engine = None

if "tts_enabled" not in st.session_state:
    st.session_state.tts_enabled = False

if "voice_input_enabled" not in st.session_state:
    st.session_state.voice_input_enabled = False

if "last_spoken_message_hash" not in st.session_state:
    st.session_state.last_spoken_message_hash = None

if "is_paused" not in st.session_state:
    st.session_state.is_paused = False

if "last_spoken_text" not in st.session_state:
    st.session_state.last_spoken_text = ""

# --- TTS Engine Initialization ---
def initialize_tts():
    """Initializes the pyttsx3 engine."""
    if st.session_state.tts_engine is None:
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 175)
            st.session_state.tts_engine = engine
            return True
        except Exception as e:
            st.error(f"Error initializing TTS engine: {e}")
            st.session_state.tts_engine = None
            return False
    return True

def speak(text):
    """Speaks the given text if TTS is enabled and not paused."""
    if not text or not st.session_state.tts_enabled or st.session_state.is_paused:
        return
        
    if initialize_tts():
        try:
            # Save the text we're about to speak
            st.session_state.last_spoken_text = text
            
            # Speak the text
            st.session_state.tts_engine.say(text)
            st.session_state.tts_engine.runAndWait()
        except Exception as e:
            st.warning(f"Could not speak text: {e}")
            # Reset engine on error
            st.session_state.tts_engine = None

def toggle_pause_resume():
    """Toggles between pause and resume states for speech."""
    if initialize_tts():
        try:
            if st.session_state.is_paused:
                # Resume speech - replay the last spoken text
                st.session_state.is_paused = False
                if st.session_state.last_spoken_text:
                    # Reinitialize to ensure we have a clean state
                    st.session_state.tts_engine = None
                    initialize_tts()
                    speak(st.session_state.last_spoken_text)
                    st.success("▶️ Resumed speech.")
                else:
                    st.info("No previous speech to resume.")
            else:
                # Pause speech
                st.session_state.tts_engine.stop()
                st.session_state.is_paused = True
                st.success("⏸️ Speech paused.")
        except Exception as e:
            st.warning(f"Could not toggle pause/resume: {e}")

# === Voice Input Function ===
def recognize_speech():
    """Listens for audio and uses Vosk for offline speech-to-text transcription."""
    recognizer = sr.Recognizer()
    st.info("🎤 Listening... Speak now.")

    if not os.path.exists(VOSK_MODEL_PATH):
        st.error(f"Vosk model not found at {VOSK_MODEL_PATH}. Please download a model from https://alphacephei.com/vosk/models.")
        return None

    try:
        model = Model(VOSK_MODEL_PATH)
        rec = KaldiRecognizer(model, 16000)

        with sr.Microphone(sample_rate=16000) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=15)

        st.info("Transcribing...")

        audio_data = audio.get_wav_data(convert_rate=16000, convert_width=2)
        wf = wave.open("temp_audio.wav", "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_data)
        wf.close()

        with wave.open("temp_audio.wav", "rb") as wf:
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                rec.AcceptWaveform(data)

        result = rec.FinalResult()
        text = json.loads(result).get("text", "")

        try:
            os.remove("temp_audio.wav")
        except OSError:
            pass

        if text:
            st.success("Transcription complete.")
            return text
        else:
            st.warning("No speech detected or transcription failed.")
            return None

    except sr.WaitTimeoutError:
        st.warning("No speech detected within timeout.")
    except sr.UnknownValueError:
        st.warning("Could not understand audio.")
    except Exception as e:
        st.error(f"An error occurred during transcription: {e}")
    finally:
        if os.path.exists("temp_audio.wav"):
            try:
                os.remove("temp_audio.wav")
            except OSError:
                pass
    return None

# === Document Processing ===
@st.cache_resource(ttl=3600)
def process_documents(files_data_with_hashes, _embedding_model_name):
    """Process multiple PDF documents, create FAISS index, and return retriever."""
    st.info(f"Processing {len(files_data_with_hashes)} PDF document(s)...")
    all_docs = []
    processed_file_names = []

    available_models = get_available_ollama_models()
    if _embedding_model_name not in available_models:
        st.error(f"Embedding model '{_embedding_model_name}' not found locally in Ollama. Please run `ollama pull {_embedding_model_name}`.")
        return None, []

    try:
        for file_info in files_data_with_hashes:
            file_content = file_info["content"]
            file_name = file_info["name"]
            file_hash = file_info["hash"]
            temp_path = os.path.join(TEMP_DIR, f"doc_{file_hash}_{int(time.time())}.pdf")

            try:
                with open(temp_path, "wb") as f:
                    f.write(file_content)

                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                all_docs.extend(docs)
                processed_file_names.append(file_name)
            except Exception as e:
                st.warning(f"Could not process file '{file_name}': {e}. Skipping this file.")
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError as e:
                        st.warning(f"Could not remove temporary file {temp_path}: {e}")

    except Exception as e:
        st.error(f"An error occurred during document loading: {e}")
        return None, []

    if not all_docs:
        st.warning("No content extracted from the uploaded document(s).")
        return None, []

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(all_docs)
    except Exception as e:
        st.error(f"Error splitting documents into chunks: {e}")
        return None, processed_file_names

    if not chunks:
        st.warning("Could not split documents into chunks.")
        return None, processed_file_names

    st.info(f"Creating embeddings using '{_embedding_model_name}' for {len(chunks)} chunks...")
    try:
        embeddings = OllamaEmbeddings(model=_embedding_model_name)
        vector_store = FAISS.from_documents(chunks, embeddings)
        st.success(f"Documents processed and indexed ({len(processed_file_names)} file(s))!")
        return vector_store.as_retriever(search_kwargs={"k": 5}), processed_file_names

    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None, processed_file_names

# === Sidebar ===
with st.sidebar:
    st.subheader("🛠️ Options")
    st.subheader("🔊 Voice Options")
    st.session_state.voice_input_enabled = st.checkbox(
        "🎙️ Enable Voice Input",
        value=st.session_state.voice_input_enabled,
        help="Use your microphone for input instead of typing."
    )
    st.session_state.tts_enabled = st.checkbox(
        "🗣️ Enable Voice Output",
        value=st.session_state.tts_enabled,
        help="The AI will speak its responses."
    )

    if st.session_state.tts_enabled:
        initialize_tts()

    # Voice control button - Pause/Resume toggle
    pause_resume_text = "⏸️ Pause" if not st.session_state.is_paused else "▶️ Resume"
    if st.button(pause_resume_text, key="pause_resume_button", on_click=toggle_pause_resume):
        pass  # The on_click handler will run

    st.markdown("---")
    st.subheader("📄 Knowledge Base (PDF)")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.file_uploader_key_counter}"
    )

    if uploaded_files:
        current_files_data = []
        hasher = hashlib.md5()
        file_names_for_display = []
        sorted_files = sorted(uploaded_files, key=lambda f: f.name)

        for uploaded_file in sorted_files:
            file_content = uploaded_file.read()
            file_hash = hashlib.md5(file_content).hexdigest()
            hasher.update(file_hash.encode('utf-8'))
            current_files_data.append({
                "content": file_content,
                "name": uploaded_file.name,
                "hash": file_hash
            })
            file_names_for_display.append(uploaded_file.name)

        current_files_hash = hasher.hexdigest()

        if current_files_hash != st.session_state.get("processed_files_hash"):
            st.session_state.processed_files_hash = current_files_hash
            retriever, processed_names = process_documents(current_files_data, DEFAULT_EMBEDDING_MODEL)
            st.session_state.retriever = retriever
            st.session_state.processed_file_names = processed_names
            st.session_state.last_spoken_message_hash = None
            st.rerun()

        elif st.session_state.retriever is None:
            st.warning("Retrying document processing...")
            retriever, processed_names = process_documents(current_files_data, DEFAULT_EMBEDDING_MODEL)
            st.session_state.retriever = retriever
            st.session_state.processed_file_names = processed_names
            if st.session_state.retriever:
                st.session_state.last_spoken_message_hash = None
                st.rerun()
            else:
                st.error("Document processing failed again.")

    if st.session_state.get("retriever"):
        processed_file_list = st.session_state.get("processed_file_names", [])
        if processed_file_list:
            display_names = ", ".join([f"'{name}'" for name in processed_file_list])
            st.success(f"✅ Ready to chat about {len(processed_file_list)} document(s): {display_names}")
        else:
            st.warning("⚠️ Documents uploaded, but none could be processed successfully.")
    elif uploaded_files and not st.session_state.get("retriever"):
        st.warning("⚠️ Document processing failed. Check logs or ensure Ollama is running and models are pulled.")
    else:
        st.info("Upload PDF document(s) to enable context-aware chat.")

    if st.button("🗑️ Clear Uploaded PDFs"):
        st.session_state.retriever = None
        st.session_state.processed_files_hash = None
        st.session_state.processed_file_names = []
        st.session_state.file_uploader_key_counter += 1
        st.session_state.last_spoken_message_hash = None
        if os.path.exists(TEMP_DIR):
            try:
                shutil.rmtree(TEMP_DIR)
                os.makedirs(TEMP_DIR, exist_ok=True)
            except Exception as e:
                st.warning(f"Could not clean up temporary directory {TEMP_DIR}: {e}")
        st.success("Uploaded PDFs cleared.")
        st.rerun()

    st.markdown("---")
    st.subheader("💬 Chat Management")

    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.session_state.last_spoken_message_hash = None
        history_filepath = get_history_filepath()
        if os.path.exists(history_filepath):
            try:
                os.remove(history_filepath)
            except OSError as e:
                st.warning(f"Could not delete chat history file: {e}")
        st.success("Chat history cleared.")
        st.rerun()

    if st.button("💾 Save Current Chat"):
        st.session_state.save_triggered = True
        save_history(st.session_state.messages)
        st.success("Chat history saved!")

# === Display Chat History ===
for msg in st.session_state.messages:
    avatar = "🦖" if isinstance(msg, HumanMessage) else "🤖"
    with st.chat_message(msg.type):
        st.write(msg.content)

# === Handle User Input ===
user_input = None
text_input_placeholder = "Ask a question about the document(s)..."

input_container = st.container()

with input_container:
    if st.session_state.voice_input_enabled:
        if st.button("🎤 Click to Speak", key="speak_button"):
            user_input = recognize_speech()
        text_input = st.chat_input("Or type your message here...", key="text_chat_input")
    else:
        text_input = st.chat_input(text_input_placeholder, key="text_chat_input_no_voice")

    if user_input is None and text_input:
        user_input = text_input

# === Process Input and Generate Response ===
if user_input:
    if not st.session_state.llm:
        st.error(f"LLM ({DEFAULT_LLM_MODEL}) is not initialized. Please ensure Ollama is running and the model is pulled.")
        user_input = None
    elif not st.session_state.retriever and uploaded_files:
        st.error("Documents were uploaded, but the retriever is not ready. Processing may have failed. Check sidebar messages.")
        user_input = None
    elif not st.session_state.retriever:
        st.warning("Please upload PDF document(s) first to chat about their content.")
        user_input = None
    else:
        st.session_state.messages.append(HumanMessage(content=user_input))

        context_text = ""
        if st.session_state.retriever:
            with st.spinner("Searching documents..."):
                try:
                    retrieved_docs = st.session_state.retriever.invoke(user_input)
                    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
                except Exception as e:
                    st.error(f"Error retrieving documents: {e}")
                    context_text = "Error retrieving context."

        messages_for_llm = []
        messages_for_llm.append(SystemMessage(content=DEFAULT_SYSTEM_PROMPT))
        history_for_llm = [msg for msg in st.session_state.messages[:-1] if isinstance(msg, (HumanMessage, AIMessage))]
        messages_for_llm.extend(history_for_llm)

        if context_text and context_text != "Error retrieving context.":
            user_input_with_context = f"Based on the following context:\n\n<context>\n{context_text}\n</context>\n\nAnswer this question: {user_input}"
            messages_for_llm.append(HumanMessage(content=user_input_with_context))
        else:
            messages_for_llm.append(HumanMessage(content=user_input))

        with st.spinner("Thinking..."):
            start_time = time.time()
            try:
                response = st.session_state.llm.invoke(messages_for_llm)
                ai_response_content = response.content
            except Exception as e:
                st.error(f"Error invoking LLM ({DEFAULT_LLM_MODEL}): {e}")
                ai_response_content = "Sorry, I encountered an error while generating a response."
            end_time = time.time()

        ai_message = AIMessage(content=ai_response_content)
        st.session_state.messages.append(ai_message)

        save_history(st.session_state.messages)
        st.rerun()

# === Speak the Last AI Message ===
if st.session_state.tts_enabled and st.session_state.messages and not st.session_state.is_paused:
    last_message = st.session_state.messages[-1]
    if isinstance(last_message, AIMessage):
        message_hash = hashlib.md5(last_message.content.encode("utf-8")).hexdigest()
        if st.session_state.get("last_spoken_message_hash") != message_hash:
            speak(last_message.content)
            st.session_state.last_spoken_message_hash = message_hash
elif not st.session_state.tts_enabled and st.session_state.get("last_spoken_message_hash") is not None:
    st.session_state.last_spoken_message_hash = None
    st.session_state.is_paused = False