# 📚 Local RAG Chatbot with Voice (Complete Offline)

This project is a local Retrieval Augmented Generation (RAG) chatbot application built with Streamlit, LangChain, and Ollama. It allows you to upload multiple PDF documents and chat with an AI model that can answer questions based on the content of those documents. The application features both voice input (speech-to-text) and output (text-to-speech) capabilities, all running completely offline.

## ✨ Features

* **Complete Offline Operation:** Runs entirely on your local machine using Ollama and Vosk
* **Multi-PDF Support:** Upload and query information from multiple PDF files simultaneously
* **Contextual Answers:** AI responds based only on the content of uploaded documents
* **Offline Voice Input:** Speak your questions using Vosk for speech recognition (no internet required)
* **Offline Voice Output:** AI responses can be spoken aloud using pyttsx3
* **Chat History Persistence:** Saves and loads your chat history locally
* **Flexible Configuration:** Customize models, prompts, and behavior through the code
* **Easy Management:** Clear buttons for uploaded PDFs and chat history

## 📋 Prerequisites

Before you begin, ensure you have the following:

1. **Python 3.8+:** [python.org/downloads](https://www.python.org/downloads/)
2. **Ollama:** [ollama.com/download](https://ollama.com/download) (must be running)
3. **Required Models:** Run these commands after installing Ollama:
  ```bash
    ollama pull mistral:7b
    ollama pull nomic-embed-text:latest
   ```
4. **Vosk Model:** Download the small English model from alphacephei.com/vosk/models and place it in a vosk-models directory   
5. **Microphone and Speakers:** Required for voice features

## 🚀 Installation & Setup

1. Clone or download the repository
2. Create and activate a virtual environment:
```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/Mac
    .venv\Scripts\activate     # Windows
```
3. Install dependencies:

```bash
   pip install -r requirements.txt
```

4. Set up Vosk models:
    Download the small English model from alphacephei.com/vosk/models
    Extract it to vosk-models/vosk-model-small-en-us-0.15 in your project directory


## ▶️ Running the Application
Ensure Ollama is running in the background

Start the application:

``` bash
streamlit run app.py
The app will open in your default browser at http://localhost:8501
```

## 📄 Usage Guide

### Basic Operation
*  Upload PDFs via the sidebar
* Wait for processing to complete (you'll see a confirmation message)
* Ask questions about the document content:
* Type in the chat box, OR
* Click the microphone button to speak your question

### Voice Features

* Enable/disable voice input and output via the sidebar checkboxes
* For voice input, click the microphone button and speak clearly
* Voice output will automatically read the AI's responses when enabled

### Managing Content
* Clear Uploaded PDFs: Removes all documents from the knowledge base
* Clear Chat History: Starts a new conversation
* Save Current Chat: Manually persists the conversation to disk

## ⚙️ Configuration
Modify these constants in app.py to customize behavior:

```bash
python
DEFAULT_LLM_MODEL = "mistral:7b"               # Change the language model
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:latest"  # Change embedding model
DEFAULT_SYSTEM_PROMPT = "..."                   # Customize AI personality
VOSK_MODEL_PATH = "vosk-models/..."             # Path to Vosk model
```

🐞 Troubleshooting

## Common Issues
Ollama Connection Problems:
Ensure Ollama is running (ollama serve)
Verify models are downloaded (ollama list)

## Voice Recognition Issues:

Check microphone permissions
Ensure Vosk model is in the correct location
Reduce background noise

## PDF Processing Errors:

Try different PDF files
Check for password protection or corruption

## TTS Problems:

On Linux, install libespeak1:

```bash
sudo apt-get install libespeak1
Check your audio output settings
```

For additional help, check the terminal output when running the app as it often contains detailed error messages.


