import os
import json
import hashlib
import time
import streamlit as st
import shutil
import pandas as pd
from PIL import Image
import requests
import logging
import torch
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration
import speech_recognition as sr
import pyttsx3
import ollama
from vosk import Model, KaldiRecognizer
import wave

# === Configuration ===
DEFAULT_LLM_MODEL = "gemma3:latest"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_SYSTEM_PROMPT = """You are Chatbot, a helpful, witty, and slightly sarcastic assistant. Answer concisely based on the provided context from documents, images, and tables. When referencing figures or tables, use the format [Figure X] or [Table Y] where X/Y are numbers starting from 1. When mentioning file names, use the exact file name as provided in the context, preserving case and spacing (e.g., 'Image1.jpg' not 'image1.jpg' or 'Image 1.jpg'). Ensure you include [Table Y] references for any relevant table data, even if the query doesn't explicitly ask for it, if it improves the answer. If the answer isn't in the context, say you don't know."""
HISTORY_DIR = "chat_history"
TEMP_DIR = "temp_docs"
TEXT_FOLDER = "data/text_documents"
IMAGE_FOLDER = "data/images"
TABLE_FOLDER = "data/tables"
DB_PATH = "data/chroma_db"
VOSK_MODEL_PATH = "vosk-models/vosk-model-small-en-us-0.15"
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Create necessary directories
for directory in [HISTORY_DIR, TEMP_DIR, TEXT_FOLDER, IMAGE_FOLDER, TABLE_FOLDER, DB_PATH]:
    os.makedirs(directory, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# === Page Configuration ===
st.set_page_config(page_title="Multi-Modal RAG Chatbot with Voice", page_icon="🤖", layout="wide")
st.title("🤖 Multi-Modal RAG Chatbot with Voice")

# === Model Loading ===
@st.cache_resource
def load_models():
    try:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", 
            torch_dtype=torch.float32
        ).to('cpu').eval()
        return embedding_model, processor, blip_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        logger.error(f"Error loading models: {str(e)}", exc_info=True)
        raise e

# === ChromaDB Initialization ===
@st.cache_resource
def init_db():
    try:
        client = PersistentClient(path=DB_PATH)
        try:
            collection = client.get_collection("documents")
            logger.info(f"Found existing collection with {collection.count()} documents")
        except Exception:
            logger.info("Creating new collection")
            collection = client.create_collection("documents")
        return collection
    except Exception as e:
        st.error(f"Error initializing ChromaDB: {str(e)}")
        raise e

# === Chat History Management ===
def get_history_filepath():
    session_id = st.session_state.get("session_id", "current_chat")
    return os.path.join(HISTORY_DIR, f"{session_id}.json")

def load_history():
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
    filepath = get_history_filepath()
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            serializable_messages = [
                {"type": "human" if isinstance(msg, HumanMessage) else "ai", "content": msg.content}
                for msg in messages if isinstance(msg, (HumanMessage, AIMessage))
            ]
            json.dump(serializable_messages, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Error saving chat history: {e}")

# === Ollama Functions ===
def get_available_ollama_models():
    try:
        model_list = ollama.list()
        return [model['name'] for model in model_list.get('models', [])]
    except Exception:
        return [DEFAULT_LLM_MODEL]

def check_ollama_status():
    try:
        health_response = requests.get("http://localhost:11434", timeout=5)
        if health_response.status_code != 200:
            return False, []
        api_response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if api_response.status_code == 200:
            return True, [model["name"] for model in api_response.json().get("models", [])]
        return False, []
    except Exception as e:
        st.error(f"Ollama connection error: {str(e)}")
        return False, []

def generate_with_ollama(prompt, model=DEFAULT_LLM_MODEL, max_tokens=1000):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens}
    }
    try:
        health_response = requests.get("http://localhost:11434", timeout=5)
        if health_response.status_code != 200:
            return "Ollama service is not running. Please start Ollama first."
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("response", "No response generated")
    except requests.exceptions.ConnectionError:
        return "Could not connect to Ollama. Please make sure Ollama is running."
    except requests.exceptions.Timeout:
        return "Request timed out. Ollama might be busy or not responding."
    except Exception as e:
        return f"Error generating response: {str(e)}"

# === Voice Functions ===
def initialize_tts():
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
    if not text or not st.session_state.tts_enabled or st.session_state.is_paused:
        return
    if initialize_tts():
        try:
            st.session_state.last_spoken_text = text
            st.session_state.tts_engine.say(text)
            st.session_state.tts_engine.runAndWait()
        except Exception as e:
            st.warning(f"Could not speak text: {e}")
            st.session_state.tts_engine = None

def toggle_pause_resume():
    if initialize_tts():
        try:
            if st.session_state.is_paused:
                st.session_state.is_paused = False
                if st.session_state.last_spoken_text:
                    st.session_state.tts_engine = None
                    initialize_tts()
                    speak(st.session_state.last_spoken_text)
                    st.success("▶️ Resumed speech.")
                else:
                    st.info("No previous speech to resume.")
            else:
                st.session_state.tts_engine.stop()
                st.session_state.is_paused = True
                st.success("⏸️ Speech paused.")
        except Exception as e:
            st.warning(f"Could not toggle pause/resume: {e}")

def recognize_speech():
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

# === Document Indexing Functions ===
def normalize_filename(filename):
    return re.sub(r'\s+', '_', filename.lower())

def index_text_documents(collection, embedding_model):
    count = 0
    if not os.path.exists(TEXT_FOLDER):
        return count
    files = os.listdir(TEXT_FOLDER)
    for filename in files:
        if filename.lower().endswith('.txt'):
            file_path = os.path.join(TEXT_FOLDER, filename)
            try:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    try:
                        with open(file_path, 'r', encoding='cp1252') as f:
                            content = f.read()
                    except UnicodeDecodeError:
                        with open(file_path, 'r', encoding='iso-8859-1') as f:
                            content = f.read()
                if not content.strip():
                    continue
                chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue
                    try:
                        embedding = embedding_model.encode(chunk).tolist()
                        collection.add(
                            documents=[chunk],
                            embeddings=[embedding],
                            metadatas=[{
                                'type': 'text',
                                'source': normalize_filename(filename),
                                'chunk': i
                            }],
                            ids=[f"txt_{filename}_{i}"]
                        )
                        count += 1
                    except Exception as e:
                        st.warning(f"Error processing chunk {i} of {filename}: {str(e)}")
            except Exception as e:
                st.error(f"Error processing {filename}: {str(e)}")
    return count

def index_images(collection, embedding_model, processor, blip_model):
    count = 0
    if not os.path.exists(IMAGE_FOLDER):
        return count
    files = os.listdir(IMAGE_FOLDER)
    for idx, filename in enumerate(files):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join(IMAGE_FOLDER, filename)
            try:
                image = Image.open(file_path).convert('RGB')
                inputs = processor(image, return_tensors="pt").to('cpu')
                with torch.no_grad():
                    outputs = blip_model.generate(**inputs)
                caption = processor.decode(outputs[0], skip_special_tokens=True)
                embedding = embedding_model.encode(caption).tolist()
                collection.add(
                    documents=[caption],
                    embeddings=[embedding],
                    metadatas=[{
                        'type': 'image',
                        'path': file_path,
                        'source': normalize_filename(filename),
                        'index': idx
                    }],
                    ids=[f"img_{filename}"]
                )
                count += 1
            except Exception as e:
                st.error(f"Error processing image {filename}: {str(e)}")
    return count

def index_tables(collection, embedding_model):
    count = 0
    if not os.path.exists(TABLE_FOLDER):
        return count
    files = os.listdir(TABLE_FOLDER)
    for idx, filename in enumerate(files):
        if filename.lower().endswith('.csv'):
            file_path = os.path.join(TABLE_FOLDER, filename)
            try:
                df = pd.read_csv(file_path)
                num_rows, num_cols = df.shape
                columns = ", ".join(df.columns.tolist())
                sample_data = str(df.head(3))
                key_values = " ".join(df.iloc[:, 0].head(5).astype(str).tolist())
                description = f"Table {filename} with {num_rows} rows and {num_cols} columns. Columns: {columns}. Key values: {key_values}"
                full_text = f"{description}\n\nSample data:\n{sample_data}"
                embedding = embedding_model.encode(full_text).tolist()
                collection.add(
                    documents=[full_text],
                    embeddings=[embedding],
                    metadatas=[{
                        'type': 'table',
                        'path': file_path,
                        'source': normalize_filename(filename),
                        'index': idx
                    }],
                    ids=[f"tbl_{filename}"]
                )
                count += 1
            except Exception as e:
                st.error(f"Error processing table {filename}: {str(e)}")
    return count

def index_documents(collection, embedding_model, processor, blip_model):
    st.info("Starting document indexing...")
    text_count = index_text_documents(collection, embedding_model)
    image_count = index_images(collection, embedding_model, processor, blip_model)
    table_count = index_tables(collection, embedding_model)
    return text_count, image_count, table_count

# === Context Retrieval ===
def retrieve_context(collection, embedding_model, query, n_results=5):
    try:
        query_embedding = embedding_model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={'type': 'table'}
        )
        logger.info(f"Table query results: {results}")
        general_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        context = {"text": [], "images": [], "tables": []}
        if results and 'documents' in results and len(results['documents']) > 0:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                context['tables'].append({
                    'description': doc,
                    'path': metadata.get('path', '')
                })
        if general_results and 'documents' in general_results and len(general_results['documents']) > 0:
            for i, doc in enumerate(general_results['documents'][0]):
                metadata = general_results['metadatas'][0][i]
                doc_type = metadata.get('type', 'text')
                if doc_type == 'text':
                    context['text'].append(doc)
                elif doc_type == 'image':
                    context['images'].append({
                        'caption': doc,
                        'path': metadata.get('path', '')
                    })
        return context
    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}")
        st.error(f"Error retrieving context: {str(e)}")
        return {"text": [], "images": [], "tables": []}

# === Response Generation ===
def generate_response_with_history(query, collection, embedding_model, chat_history):
    context = retrieve_context(collection, embedding_model, query)
    history_text = ""
    if chat_history:
        recent_history = chat_history[-6:]
        for msg in recent_history:
            if isinstance(msg, HumanMessage):
                history_text += f"Human: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_text += f"Assistant: {msg.content}\n"
    prompt = f"""You are a helpful assistant. Use the provided context to answer questions accurately.
When referencing figures or tables, use the format [Figure X] or [Table Y] where X/Y are numbers starting from 1. Ensure you include [Table Y] references for any relevant table data, even if the query doesn't explicitly ask for it, if it improves the answer. Use exact file names as provided in the context, preserving case and spacing.

Previous conversation:
{history_text}

Current question: {query}

Text Context: {' '.join(context['text'])}

Image Context: {' '.join([img['caption'] for img in context['images']])}

Table Context: {' '.join([tbl['description'] for tbl in context['tables']])}

Answer concisely and include [Figure X] or [Table Y] references when appropriate:"""
    response = generate_with_ollama(prompt)
    return response, context

# === Display Functions ===
def get_case_insensitive_path(folder, filename):
    for f in os.listdir(folder):
        if f.lower() == filename.lower():
            return os.path.join(folder, f)
    return None

def display_referenced_files(response_text, collection):
    logger.info(f"Processing response: {response_text}")
    image_metadata = collection.get(where={'type': 'image'})['metadatas']
    table_metadata = collection.get(where={'type': 'table'})['metadatas']
    logger.info(f"Image metadata: {image_metadata}")
    logger.info(f"Table metadata: {table_metadata}")
    
    normalized_response = normalize_filename(response_text)
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    table_files = [f for f in os.listdir(TABLE_FOLDER) if f.lower().endswith('.csv')]
    
    direct_image_refs = [f for f in image_files if normalize_filename(f) in normalized_response]
    direct_table_refs = [f for f in table_files if normalize_filename(f) in normalized_response]
    
    figure_refs = re.findall(r'\[Figure (\d+)\]', response_text)
    table_refs = re.findall(r'\[Table (\d+)\]', response_text)
    
    if figure_refs or direct_image_refs:
        st.subheader("Referenced Images")
        for ref in figure_refs:
            try:
                idx = int(ref) - 1
                for meta in image_metadata:
                    if meta.get('index') == idx:
                        image_path = meta.get('path')
                        if os.path.exists(image_path):
                            st.image(image_path, caption=f"Figure {ref}")
                        break
            except (ValueError, IndexError):
                continue
        for img_file in direct_image_refs:
            image_path = get_case_insensitive_path(IMAGE_FOLDER, img_file)
            if image_path and os.path.exists(image_path):
                st.image(image_path, caption=img_file)
            else:
                st.error(f"Image {img_file} not found.")
    
    if table_refs or direct_table_refs:
        st.subheader("Referenced Tables")
        for ref in table_refs:
            try:
                idx = int(ref) - 1
                for meta in table_metadata:
                    if meta.get('index') == idx:
                        table_path = meta.get('path')
                        if os.path.exists(table_path):
                            df = pd.read_csv(table_path)
                            st.caption(f"Table {ref}")
                            st.dataframe(df)
                        break
            except (ValueError, IndexError):
                continue
        for tbl_file in direct_table_refs:
            table_path = get_case_insensitive_path(TABLE_FOLDER, tbl_file)
            if table_path and os.path.exists(table_path):
                df = pd.read_csv(table_path)
                st.caption(tbl_file)
                st.dataframe(df)
            else:
                st.error(f"Table {tbl_file} not found.")

def display_chat_message(msg, context=None):
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="🦖"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.write(msg.content)
            display_referenced_files(msg.content, collection)

def handle_file_uploads():
    st.header("Upload Documents")
    text_files = st.file_uploader("Upload Text Files", type=["txt"], accept_multiple_files=True)
    if text_files:
        for uploaded_file in text_files:
            save_path = os.path.join(TEXT_FOLDER, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Saved {len(text_files)} text files")
    image_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png", "bmp", "gif"], accept_multiple_files=True)
    if image_files:
        for uploaded_file in image_files:
            save_path = os.path.join(IMAGE_FOLDER, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Saved {len(image_files)} image files")
    table_files = st.file_uploader("Upload Tables", type=["csv"], accept_multiple_files=True)
    if table_files:
        for uploaded_file in table_files:
            save_path = os.path.join(TABLE_FOLDER, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Saved {len(table_files)} table files")

def clear_database(collection):
    try:
        all_docs = collection.get()
        if all_docs and 'ids' in all_docs and all_docs['ids']:
            collection.delete(ids=all_docs['ids'])
            return len(all_docs['ids'])
        return 0
    except Exception as e:
        st.error(f"Error clearing database: {str(e)}")
        return 0

# === Initialize Session State ===
if "session_id" not in st.session_state:
    st.session_state.session_id = "current_chat"
if "messages" not in st.session_state:
    st.session_state.messages = load_history()
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
if "last_context" not in st.session_state:
    st.session_state.last_context = None
if "table_refs" not in st.session_state:
    st.session_state.table_refs = []

# === Load Models and Initialize DB ===
try:
    embedding_model, processor, blip_model = load_models()
    collection = init_db()
    st.success("✅ Models and database initialized successfully!")
except Exception as e:
    st.error(f"Failed to initialize models: {e}")
    st.stop()

# === Sidebar ===
with st.sidebar:
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
    pause_resume_text = "⏸️ Pause" if not st.session_state.is_paused else "▶️ Resume"
    if st.button(pause_resume_text, key="pause_resume_button", on_click=toggle_pause_resume):
        pass
    st.markdown("---")
    st.subheader("📄 Document Management")
    handle_file_uploads()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Re-index", type="primary"):
            with st.spinner("Indexing documents..."):
                try:
                    text_count, image_count, table_count = index_documents(collection, embedding_model, processor, blip_model)
                    st.success(f"Documents indexed! Added {text_count} text chunks, {image_count} images, and {table_count} tables.")
                except Exception as e:
                    st.error(f"Error during indexing: {str(e)}")
    with col2:
        if st.button("Clear Database", type="secondary"):
            with st.spinner("Clearing database..."):
                try:
                    cleared_count = clear_database(collection)
                    st.success(f"Database cleared! Removed {cleared_count} documents.")
                except Exception as e:
                    st.error(f"Error clearing database: {str(e)}")
    try:
        count = collection.count()
        st.markdown("### Database Info")
        st.write(f"Documents in DB: {count}")
    except Exception as e:
        st.error(f"Error getting DB count: {str(e)}")
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
        save_history(st.session_state.messages)
        st.success("Chat history saved!")

# === Display Chat History ===
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage) and st.session_state.last_context:
        display_chat_message(msg, st.session_state.last_context)
    else:
        display_chat_message(msg)

# === Handle User Input ===
user_input = None
if st.session_state.voice_input_enabled:
    if st.button("🎤 Click to Speak", key="speak_button"):
        user_input = recognize_speech()
    text_input = st.chat_input("Or type your message here...", key="text_chat_input")
else:
    text_input = st.chat_input("Ask a question about your documents...", key="text_chat_input_no_voice")
if user_input is None and text_input:
    user_input = text_input

# === Process Input and Generate Response ===
if user_input:
    ollama_connected, _ = check_ollama_status()
    if not ollama_connected:
        st.error("Cannot generate response: Ollama is not connected. Please make sure Ollama is running.")
    else:
        try:
            doc_count = collection.count()
            if doc_count == 0:
                st.warning("No documents found in the database. Please upload and index some documents first.")
            else:
                st.session_state.messages.append(HumanMessage(content=user_input))
                with st.spinner("Generating response..."):
                    response, context = generate_response_with_history(
                        user_input, collection, embedding_model, st.session_state.messages[:-1]
                    )
                    st.session_state.last_context = context
                    st.session_state.table_refs = [tbl['path'] for tbl in context['tables']]
                ai_message = AIMessage(content=response)
                st.session_state.messages.append(ai_message)
                save_history(st.session_state.messages)
                if st.session_state.table_refs:
                    st.subheader("Referenced Tables from Context")
                    for idx, table_path in enumerate(st.session_state.table_refs, 1):
                        if os.path.exists(table_path):
                            df = pd.read_csv(table_path)
                            st.caption(f"Table {idx}")
                            st.dataframe(df)
                st.rerun()
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

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