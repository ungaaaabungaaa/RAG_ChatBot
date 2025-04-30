# 📚 Sammy's RAG ChatBot (Multi-PDF)

This project is a local Retrieval Augmented Generation (RAG) chatbot application built with Streamlit, LangChain, and Ollama. It allows you to upload multiple PDF documents and chat with an AI model that can answer questions based *only* on the content of those documents. It also includes optional voice input and output features.

The chatbot is named "Sammy" and is designed to be helpful, witty, and slightly sarcastic, adhering strictly to the provided document context.

## ✨ Features

* **Local RAG:** Runs entirely on your local machine using Ollama.
* **Multi-PDF Support:** Upload and query information from multiple PDF files simultaneously.
* **Contextual Answers:** The AI answers based *only* on the content of the uploaded documents.
* **Chat History:** Saves and loads your chat history locally.
* **Voice Input (STT):** Speak your questions using your microphone (requires SpeechRecognition).
* **Voice Output (TTS):** The AI's responses can be spoken aloud (requires pyttsx3).
* **Clear Options:** Buttons to easily clear uploaded PDFs and chat history.

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python 3.8+:** You can download Python from [python.org](https://www.python.org/downloads/).
2.  **Ollama:** Install Ollama from [ollama.com](https://ollama.com/download). Make sure the Ollama service is running in the background.
3.  **Required Ollama Models:** Pull the models used by the application. Open your terminal or command prompt and run:
    ```bash
    ollama pull mistral:7b
    ollama pull nomic-embed-text:latest
    ```
    *(Note: These are the default models hardcoded in the script. You can change them by editing the `app.py` file, but ensure you pull the models you intend to use).*
4.  **Microphone and Speakers:** Necessary if you plan to use the voice input/output features.
5.  **Additional dependency for TTS on Linux:** If you are running on Linux, you might need `libespeak1`. Install it via your distribution's package manager (e.g., `sudo apt-get install libespeak1` on Debian/Ubuntu).

## 🚀 Installation

Follow these steps to set up and run the application. It is highly recommended to use a virtual environment.

1.  **Save the Script:** Save the provided Python code as `app.py` in a directory of your choice.

2.  **Open Terminal/Command Prompt:** Navigate to the directory where you saved `app.py`.

3.  **Create a Virtual Environment:**
    Using `venv` (recommended, included with Python 3.3+):

    * **macOS and Linux:**
        ```bash
        python3 -m venv .venv
        ```
    * **Windows:**
        ```bash
        python -m venv .venv
        ```
    This creates a directory named `.venv` containing the virtual environment.

4.  **Activate the Virtual Environment:**
    * **macOS and Linux:*
        ```bash
        source .venv/bin/activate
        ```
    * **Windows:**
        ```bash
        .venv\Scripts\activate
        ```
    You should see `(.venv)` or similar at the start of your terminal prompt, indicating the virtual environment is active.

5.  **Create `requirements.txt`:** Create a file named `requirements.txt` in the same directory as `app.py` and add the following lines:

    ```plaintext
    streamlit
    langchain-ollama
    langchain-community
    langchain-core
    langchain # Although core/community are used, this might be needed for base imports
    pypdf
    faiss-cpu
    SpeechRecognition
    pyttsx3
    ollama # The ollama python client library
    ```

6.  **Install Dependencies:** With the virtual environment activated, install the required libraries using pip:

    ```bash
    pip install -r requirements.txt
    ```
    *(Note: `faiss-cpu` is for CPU-only FAISS. If you have a compatible GPU and CUDA set up, you might consider `faiss-gpu`, but `faiss-cpu` is sufficient for most local uses).*

7.  **Deactivate (Optional):** When you are done working on the project, you can deactivate the virtual environment by simply typing:

    ```bash
    deactivate
    ```

## ▶️ Running the Application

1.  **Activate your virtual environment** (if not already active):
    * **macOS and Linux:** `source .venv/bin/activate`
    * **Windows:** `.venv\Scripts\activate`

2.  **Ensure Ollama is running** and the required models (`mistral:7b`, `nomic-embed-text:latest`) are pulled (`ollama list` to check).

3.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

4.  Your web browser should open automatically to the application interface. If not, open your browser and go to the address shown in the terminal (usually `http://localhost:8501`).

## 📄 Usage

1.  The application will open in your web browser.
2.  Use the sidebar on the left to interact:
    * **Upload PDF files:** Drag and drop or click to upload one or more PDF documents.
    * Wait for the "Document processing..." and "Creating embeddings..." steps to complete. A success message will appear in the sidebar when it's ready.
    * **Enable Voice Input/Output:** Check the boxes in the sidebar to enable these features. If enabling voice input, a "Click to Speak" button will appear below the chat input.
    * **Clear PDFs:** Removes the uploaded documents and resets the knowledge base.
    * **Clear Chat History:** Clears the current conversation from the display and the saved history file.
    * **Save Current Chat:** Manually saves the current conversation to the history file.
3.  Once documents are processed, type your question related to the document content in the chat input box at the bottom of the page and press Enter, or click the "Click to Speak" button (if enabled) and speak your query.
4.  Sammy will provide a response based on the uploaded documents.
5.  If voice output is enabled, the AI's response will be spoken aloud after it appears.

## ⚙️ Configuration & Customization

Most configuration is done by editing the `app.py` file directly.

### Changing Titles and Icons

Streamlit handles the browser tab title and favicon (the small icon in the browser tab). It does *not* directly control the native application window title or taskbar/dock icon when run with `streamlit run`. To customize the native application icon, you would typically need to package your application into an executable using tools like PyInstaller, which is beyond the scope of this basic setup.

Here's how to change the browser tab title and favicon using `st.set_page_config`:

1.  Open the `app.py` file in a text editor.
2.  Find the line:
    ```python
    st.set_page_config(page_title="Local RAG Chatbot", page_icon="📚", layout="wide")
    ```
3.  **`page_title`:** Change the string `"Local RAG Chatbot"` to your desired browser tab title.
4.  **`page_icon`:**
    * You can use a short emoji string (like `"📚"`, `"🤖"`, `"💬"`).
    * Alternatively, you can use the URL of an image file (e.g., `page_icon="http://example.com/my_icon.png"`) or the path to a local image file (e.g., `page_icon="images/my_icon.png"` - ensure the path is correct relative to where you run the app).
    * Common image formats like `.png`, `.jpg`, `.svg` are usually supported.
5.  Save the `app.py` file.
6.  Restart the Streamlit application (`streamlit run app.py`) to see the changes.

### Other Hardcoded Configurations

You can also edit `app.py` to change:

* `DEFAULT_LLM_MODEL`: Change `"mistral:7b"` to another available Ollama language model.
* `DEFAULT_EMBEDDING_MODEL`: Change `"nomic-embed-text:latest"` to another available Ollama embedding model.
* `DEFAULT_SYSTEM_PROMPT`: Modify the system prompt to change the AI's persona and instructions.
* `HISTORY_DIR`: Change the directory where chat history is saved.
* `TEMP_DIR`: Change the directory used for temporary PDF files.

Remember to save the file and restart the Streamlit application after making any changes.

## 🐞 Troubleshooting

* **"FileNotFoundError: [Errno 2] No such file or directory: 'chat_history/current_chat.json'"**: This is usually harmless on the first run as the directory and file are created. If it persists, check permissions for the directory where `app.py` is located.
* **"Could not connect to Ollama..." or "LLM is not initialized..."**:
    * Ensure Ollama is installed and running in the background.
    * Check the Ollama logs for errors.
    * Verify that the `DEFAULT_LLM_MODEL` and `DEFAULT_EMBEDDING_MODEL` specified in `app.py` are actually pulled (`ollama list`). If not, pull them using `ollama pull <model_name>`.
* **"Error creating vector store..." or Embedding issues**:
    * Ensure the `DEFAULT_EMBEDDING_MODEL` is pulled and available in Ollama.
    * Check Ollama status.
    * The embedding process can be memory-intensive for large documents. Ensure your system has enough RAM.
* **PDF Processing Issues**:
    * If a specific PDF fails, it might be corrupted, password-protected, or in an unusual format. Try opening it in a standard PDF reader.
    * Check the terminal output for error messages from `PyPDFLoader`.
* **Voice Input Errors ("Could not understand audio", "Could not request results...")**:
    * Ensure your microphone is connected and working correctly.
    * Check your operating system's microphone privacy settings.
    * Minimize background noise.
    * Ensure you have a stable internet connection if using `recognize_google` (it's an online service). You could switch to an offline recognizer like `recognize_sphinx` if needed (requires installing `PocketSphinx`).
* **Voice Output Errors ("Error initializing TTS engine...", "Could not speak text...")**:
    * Ensure your speakers are working.
    * Check volume levels.
    * On Linux, ensure `libespeak1` is installed.
    * `pyttsx3` can sometimes have compatibility issues with specific Python versions or operating system configurations. Check the `pyttsx3` documentation or common issues online if problems persist.
* **Permissions Errors**: Ensure the user running the Streamlit app has write permissions for the directory where `app.py` is located (specifically for creating `chat_history` and `temp_docs` subdirectories and files within them).
* **Clearing Cache**: If the application behaves unexpectedly after code changes or document uploads, you can try clearing the Streamlit cache. While `st.cache_data` and `st.cache_resource` have TTLs, sometimes a manual clear helps. You can find the Streamlit cache directory (location varies by OS) and delete its contents, or look for options within the Streamlit debug menu (accessible via the "three dots" menu in the top right of the app -> "Clear cache").

If you encounter persistent issues, check the terminal where you ran `streamlit run app.py` for detailed error messages, as they often provide clues about the problem.
