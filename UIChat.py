import os
import json
import hashlib
import time # Keep time import
import streamlit as st
import shutil # Import shutil for directory removal

from langchain_ollama import ChatOllama, OllamaEmbeddings # Updated imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage # Import SystemMessage
from langchain_community.document_loaders import PyPDFLoader # Only PDF needed now
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import speech_recognition as sr
import pyttsx3
import ollama # Keep ollama for listing models

# === Configuration ===
DEFAULT_LLM_MODEL = "mistral:7b" # Hardcoded LLM Model
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:latest" # Hardcoded Embedding Model
DEFAULT_SYSTEM_PROMPT = "You are Sammy, a helpful, witty, and slightly sarcastic assistant. Answer concisely based *only* on the provided context documents. If the answer isn't in the context, say you don't know." # Hardcoded System Prompt
HISTORY_DIR = "chat_history" # Use a distinct directory
TEMP_DIR = "temp_docs" # Directory for temporary files
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True) # Create temp dir

# === Page Configuration ===
st.set_page_config(page_title="Local RAG Chatbot", page_icon="📚", layout="wide")
st.title("📚 Sammy's RAG ChatBot (Multi-PDF)")

# === Chat History Disk Persistence ===
def get_history_filepath():
    # Simple session ID for this example, could be made more dynamic
    session_id = "current_chat"
    return os.path.join(HISTORY_DIR, f"{session_id}.json")

# Use st.cache_data for loading data
@st.cache_data(ttl=300)  # Cache for 5 minutes
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
                    # Ignore system messages in history load, they are set dynamically
                return history
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            st.error(f"Error loading chat history: {e}. Starting fresh.")
            return [] # Return empty list on error
    return []

def save_history(messages):
    """Saves chat history to a JSON file."""
    filepath = get_history_filepath()
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            # Filter out potential SystemMessages before saving if they accidentally got in
            serializable_messages = [
                {"type": "human" if isinstance(msg, HumanMessage) else "ai", "content": msg.content}
                for msg in messages if isinstance(msg, (HumanMessage, AIMessage))
            ]
            json.dump(serializable_messages, f, indent=2, ensure_ascii=False)
        # Clear the trigger after saving
        if "save_triggered" in st.session_state:
            del st.session_state.save_triggered
    except Exception as e:
        st.error(f"Error saving chat history: {e}")

# === Ollama Model Listing (Still useful for checking if default models exist) ===
@st.cache_data(ttl=600) # Cache model list for 10 minutes
def get_available_ollama_models():
    """Fetches the list of locally available Ollama models."""
    try:
        model_list = ollama.list()
        return [model['name'] for model in model_list.get('models', [])]
    except Exception as e:
        # st.warning(f"Could not connect to Ollama or list models: {e}. Using default models.") # Suppress frequent warnings
        # Fallback to defaults if connection fails
        return [DEFAULT_LLM_MODEL, DEFAULT_EMBEDDING_MODEL]

# === Initialize Session State ===
if "messages" not in st.session_state:
    st.session_state.messages = load_history() # Load history on first run

if "llm" not in st.session_state:
    # Initialize LLM with the default model
    try:
        st.session_state.llm = ChatOllama(model=DEFAULT_LLM_MODEL)
        # st.info(f"Using LLM: {DEFAULT_LLM_MODEL}") # Suppress frequent info
    except Exception as e:
        st.error(f"Failed to initialize default LLM {DEFAULT_LLM_MODEL}: {e}. Ensure Ollama is running and the model is pulled.")
        st.session_state.llm = None # Set to None if failed

if "retriever" not in st.session_state:
    st.session_state.retriever = None # Will store the FAISS retriever

if "processed_files_hash" not in st.session_state:
    st.session_state.processed_files_hash = None # To track uploaded file set changes

if "processed_file_names" not in st.session_state:
     st.session_state.processed_file_names = [] # List of names of successfully processed files

if "file_uploader_key_counter" not in st.session_state:
    st.session_state.file_uploader_key_counter = 0 # Counter to reset the file uploader widget

if "tts_engine" not in st.session_state:
    st.session_state.tts_engine = None # Initialize TTS engine later if needed

if "tts_enabled" not in st.session_state:
    st.session_state.tts_enabled = False

if "voice_input_enabled" not in st.session_state:
    st.session_state.voice_input_enabled = False

# --- TTS Engine Initialization ---
def initialize_tts():
    if st.session_state.tts_engine is None:
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 175) # Set default rate
            st.session_state.tts_engine = engine
            return True
        except Exception as e:
            st.error(f"Error initializing TTS engine: {e}")
            st.session_state.tts_engine = None # Ensure it's None if failed
            return False
    return True # Already initialized

def speak(text):
    """Uses pyttsx3 to speak the given text if TTS is enabled and initialized."""
    if st.session_state.tts_enabled and st.session_state.tts_engine:
        try:
            st.session_state.tts_engine.stop() # Stop previous speech if any
            st.session_state.tts_engine.say(text)
            st.session_state.tts_engine.runAndWait()
        except Exception as e:
            st.warning(f"Could not speak text: {e}")
    elif st.session_state.tts_enabled and not st.session_state.tts_engine:
         st.warning("TTS Engine failed to initialize. Cannot speak.")

# === Voice Input Function ===
def recognize_speech():
    """Listens for audio and uses SpeechRecognition to transcribe."""
    if not st.session_state.voice_input_enabled:
        return None

    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening...")
        try:
            # Adjust for ambient noise once at the start
            # recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=15) # Add timeouts
        except sr.WaitTimeoutError:
            st.warning("No speech detected within timeout.")
            return None
        except Exception as e:
            st.error(f"Error accessing microphone: {e}")
            return None

    try:
        st.info("Transcribing...")
        text = recognizer.recognize_google(audio)
        st.success("Transcription complete.")
        return text
    except sr.UnknownValueError:
        st.warning("Could not understand audio.")
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during transcription: {e}")
    return None


# === Document Processing (Cached for multiple files) ===
# Use st.cache_resource for objects like retrievers that shouldn't be pickled
# Cache key now depends on the combined hash of file contents and the embedding model name
@st.cache_resource(ttl=3600) # Cache retriever for 1 hour for the same content
def process_documents(files_data_with_hashes, _embedding_model_name):
    """Process multiple PDF documents, create FAISS index, and return retriever."""
    st.info(f"Processing {len(files_data_with_hashes)} PDF document(s)...")
    all_docs = []
    processed_file_names = [] # Keep track of successfully processed files

    # Check if embedding model is available
    available_models = get_available_ollama_models()
    if _embedding_model_name not in available_models:
        st.error(f"Embedding model '{_embedding_model_name}' not found locally in Ollama. Please run `ollama pull {_embedding_model_name}`.")
        return None, [] # Return None retriever and empty list of names

    try:
        # 1. Load all documents from all files
        for file_info in files_data_with_hashes:
            file_content = file_info["content"]
            file_name = file_info["name"]
            file_hash = file_info["hash"] # Unique hash per file
            # Use a more robust temporary file creation
            temp_path = os.path.join(TEMP_DIR, f"doc_{file_hash}_{int(time.time())}.pdf") # Add timestamp for extra uniqueness

            try:
                with open(temp_path, "wb") as f:
                    f.write(file_content)

                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                all_docs.extend(docs)
                processed_file_names.append(file_name) # Add to success list
            except Exception as e:
                st.warning(f"Could not process file '{file_name}': {e}. Skipping this file.")
            finally:
                # Clean up temp file immediately after loading or on error
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError as e:
                         st.warning(f"Could not remove temporary file {temp_path}: {e}")


    except Exception as e:
        st.error(f"An error occurred during document loading: {e}")
        # Attempt cleanup of any remaining temp files (less likely needed here)
        return None, []

    if not all_docs:
        st.warning("No content extracted from the uploaded document(s).")
        return None, []

    # 2. Split into chunks
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(all_docs)
    except Exception as e:
        st.error(f"Error splitting documents into chunks: {e}")
        return None, processed_file_names # Return names processed so far

    if not chunks:
        st.warning("Could not split documents into chunks.")
        return None, processed_file_names

    # 3. Create embeddings and FAISS vector store
    st.info(f"Creating embeddings using '{_embedding_model_name}' for {len(chunks)} chunks...")
    try:
        embeddings = OllamaEmbeddings(model=_embedding_model_name)
        vector_store = FAISS.from_documents(chunks, embeddings)
        st.success(f"Documents processed and indexed ({len(processed_file_names)} file(s))!")
        return vector_store.as_retriever(search_kwargs={"k": 5}), processed_file_names # Return retriever and processed names

    except Exception as e:
        # This could be an Ollama connection error or embedding issue
        st.error(f"Error creating vector store: {e}")
        return None, processed_file_names

# === Sidebar ===
with st.sidebar:

    st.subheader("🛠️ Options")

    st.subheader("🔊 Voice Options")
    st.session_state.voice_input_enabled = st.checkbox("🎙️ Enable Voice Input", value=st.session_state.voice_input_enabled)
    st.session_state.tts_enabled = st.checkbox("🗣️ Enable Voice Output", value=st.session_state.tts_enabled)

    # Initialize TTS engine if enabled and not already initialized
    if st.session_state.tts_enabled:
        initialize_tts()

    st.markdown("---")
    st.subheader("📄 Knowledge Base (PDF)")
    uploaded_files = st.file_uploader( # Changed variable name
        "Upload one or more PDF files",
        type=["pdf"],         # Only PDF
        accept_multiple_files=True, # Allow multiple
        key=f"file_uploader_{st.session_state.file_uploader_key_counter}" # Use dynamic key
    )

    # Process documents if uploaded and the set of files changed
    if uploaded_files: # Check if the list is not empty
        # Calculate a combined hash for the current set of uploaded files
        current_files_data = []
        hasher = hashlib.md5()
        file_names_for_display = []
        # Sort files by name to ensure consistent hash order
        sorted_files = sorted(uploaded_files, key=lambda f: f.name)

        for uploaded_file in sorted_files:
            file_content = uploaded_file.read()
            file_hash = hashlib.md5(file_content).hexdigest()
            hasher.update(file_hash.encode('utf-8')) # Update combined hash
            current_files_data.append({
                "content": file_content,
                "name": uploaded_file.name,
                "hash": file_hash # Individual hash needed for temp naming
            })
            file_names_for_display.append(uploaded_file.name)

        current_files_hash = hasher.hexdigest()

        # Only re-process if the set of files is new or has changed
        if current_files_hash != st.session_state.get("processed_files_hash"):
            st.session_state.processed_files_hash = current_files_hash # Store the new combined hash

            # Call the cached processing function
            # Pass the list of file data and the hardcoded embedding model name
            retriever, processed_names = process_documents(current_files_data, DEFAULT_EMBEDDING_MODEL)
            st.session_state.retriever = retriever # Update retriever in session state
            st.session_state.processed_file_names = processed_names # Store names of successfully processed files

            # Optional: Clear chat history when new documents are processed
            # st.session_state.messages = [] # Decided against auto-clear for now

            st.rerun() # Rerun to update status and display

        elif st.session_state.retriever is None:
             # Handle case where files are same but retriever failed previously - retry processing
             st.warning("Retrying document processing...")
             retriever, processed_names = process_documents(current_files_data, DEFAULT_EMBEDDING_MODEL)
             st.session_state.retriever = retriever
             st.session_state.processed_file_names = processed_names
             if st.session_state.retriever: # Only rerun if successful this time
                  st.rerun()
             else:
                  st.error("Document processing failed again.")


    # Display status based on retriever and processed files
    if st.session_state.get("retriever"):
        processed_file_list = st.session_state.get("processed_file_names", [])
        if processed_file_list:
            display_names = ", ".join([f"'{name}'" for name in processed_file_list])
            st.success(f"✅ Ready to chat about {len(processed_file_list)} document(s): {display_names}")
        else:
            st.warning("⚠️ Documents uploaded, but none could be processed successfully.") # Should ideally not happen if retriever is set
    elif uploaded_files and not st.session_state.get("retriever"):
        st.warning("⚠️ Document processing failed. Check logs or ensure Ollama is running and models are pulled.")
    else:
        st.info("Upload PDF document(s) to enable context-aware chat.")

    # === Clear PDFs Button ===
    if st.button("🗑️ Clear Uploaded PDFs"):
        st.session_state.retriever = None
        st.session_state.processed_files_hash = None
        st.session_state.processed_file_names = []
        st.session_state.file_uploader_key_counter += 1 # Increment key to reset uploader widget
        # Clean up the temporary directory just in case
        if os.path.exists(TEMP_DIR):
             try:
                 shutil.rmtree(TEMP_DIR)
                 os.makedirs(TEMP_DIR, exist_ok=True) # Recreate the directory
             except Exception as e:
                 st.warning(f"Could not clean up temporary directory {TEMP_DIR}: {e}")
        st.success("Uploaded PDFs cleared.")
        st.rerun() # Rerun to update sidebar status and reset uploader

    st.markdown("---")
    st.subheader("💬 Chat Management")

    # === Clear Chat History Button ===
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        history_filepath = get_history_filepath()
        if os.path.exists(history_filepath):
            try:
                os.remove(history_filepath)
            except OSError as e:
                st.warning(f"Could not delete chat history file: {e}")
        st.success("Chat history cleared.")
        st.rerun() # Rerun to clear displayed chat messages

    # === Save Chat History Button ===
    if st.button("💾 Save Current Chat"): # Better label
        st.session_state.save_triggered = True # Set trigger
        save_history(st.session_state.messages)
        st.success("Chat history saved!")


# === Display Chat History ===
st.subheader("💬 Chat")
# Display messages from session state
for msg in st.session_state.messages:
    avatar = "👤" if isinstance(msg, HumanMessage) else "🤖"
    with st.chat_message(msg.type): # Use msg.type which is 'human' or 'ai'
        st.write(msg.content)

# === Handle User Input (Text or Voice) ===
user_input = None
# input_method is determined inside the input handling logic

# Placeholder for the chat input area logic
# This needs to be outside the sidebar
if st.session_state.voice_input_enabled:
    if st.button("🎤 Click to Speak"):
        user_input = recognize_speech()
        # No need to set input_method explicitly here, user_input being not None indicates voice
    # Provide text input as fallback or primary if button not clicked
    text_input = st.chat_input("Or type your message here...")
    if text_input and user_input is None: # Only use text if voice wasn't just used
        user_input = text_input
else:
    # Standard text input if voice is disabled
    user_input = st.chat_input("Ask a question about the document(s)...")


# === Process Input and Generate Response ===
# Move the input processing block outside the sidebar
if user_input:
    if not st.session_state.llm:
        st.error(f"LLM ({DEFAULT_LLM_MODEL}) is not initialized. Please ensure Ollama is running and the model is pulled.")
    elif not st.session_state.retriever and uploaded_files: # Files uploaded but no retriever
        st.error("Documents were uploaded, but the retriever is not ready. Processing may have failed. Check sidebar messages.")
    elif not st.session_state.retriever: # No files uploaded or processed
         st.warning("Please upload PDF document(s) first to chat about their content.")
    else:
        # 1. Add user message to history and display it (Streamlit reruns will handle the display)
        st.session_state.messages.append(HumanMessage(content=user_input))
        # Streamlit will automatically re-run and display the new message in the history loop above

        # 2. Prepare context using the retriever
        context_text = ""
        # Ensure retriever exists before invoking
        if st.session_state.retriever:
            with st.spinner("Searching documents..."):
                try:
                    retrieved_docs = st.session_state.retriever.invoke(user_input)
                    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
                except Exception as e:
                    st.error(f"Error retrieving documents: {e}")
                    context_text = "Error retrieving context." # Inform LLM about failure
        else:
            # This case should be less likely due to checks above, but good for robustness
             st.warning("Retriever not available, cannot search documents.")


        # 3. Construct message list for LLM
        messages_for_llm = []
        # Add system prompt (using the hardcoded default)
        messages_for_llm.append(SystemMessage(content=DEFAULT_SYSTEM_PROMPT))
        # Add chat history (excluding the latest user message already displayed)
        # Only include Human/AI messages from history for context
        history_for_llm = [msg for msg in st.session_state.messages[:-1] if isinstance(msg, (HumanMessage, AIMessage))]
        messages_for_llm.extend(history_for_llm)

        # Add the latest user message, potentially with context
        if context_text:
            user_input_with_context = f"Based on the following context:\n\n<context>\n{context_text}\n</context>\n\nAnswer this question: {user_input}"
            messages_for_llm.append(HumanMessage(content=user_input_with_context))
        else:
             # Add original user input if no context was retrieved or retriever failed
             messages_for_llm.append(HumanMessage(content=user_input))


        # 4. Get response from LLM
        with st.spinner("Thinking..."):
            start_time = time.time()
            try:
                response = st.session_state.llm.invoke(messages_for_llm)
                ai_response_content = response.content
            except Exception as e:
                 st.error(f"Error invoking LLM ({DEFAULT_LLM_MODEL}): {e}")
                 ai_response_content = "Sorry, I encountered an error while generating a response."

            end_time = time.time()

        # 5. Add AI response to history and display it (Streamlit reruns will handle the display)
        ai_message = AIMessage(content=ai_response_content)
        st.session_state.messages.append(ai_message)

        # Streamlit will automatically re-run and display the new message in the history loop above
        # Optionally display generation time:
        # with st.chat_message("assistant", avatar="🤖"):
        #     st.caption(f"Responded in {end_time - start_time:.2f}s") # This might show under the *last* message

        # 6. Save history (simple version saves every time)
        # Save after each turn to ensure progress isn't lost
        save_history(st.session_state.messages)


        # 7. Speak the response if enabled (handled after the message is added to state/displayed on rerun)
        # This happens on the next rerun after the new message is added.
        # A more immediate speak would require managing state slightly differently or using callbacks.
        # For simplicity, let's keep it triggering on the rerun caused by adding the message.
        # A dedicated "speak last message" state variable could ensure it only speaks once per AI message.
        # Let's add a simple check to speak the *last* AI message added.
        # This needs to be outside the input processing block, after messages are displayed.


# --- Speak the last AI message if enabled ---
# This section executes on every rerun *after* messages are potentially added
if st.session_state.tts_enabled and st.session_state.messages:
    last_message = st.session_state.messages[-1]
    # Use a session state variable to track if the last message has been spoken
    if isinstance(last_message, AIMessage) and not st.session_state.get("last_spoken_message_hash") == hashlib.md5(last_message.content.encode()).hexdigest():
         speak(last_message.content)
         st.session_state.last_spoken_message_hash = hashlib.md5(last_message.content.encode()).hexdigest()
elif not st.session_state.tts_enabled and st.session_state.get("last_spoken_message_hash"):
     # Clear the hash if TTS is disabled
     del st.session_state.last_spoken_message_hash