# 🧠 Personal AI Chatbot with Document Q&A and Voice Support

A fully local, privacy-focused AI chatbot built with **LangChain**, **Ollama**, and **Streamlit**, designed to function as a personal assistant with voice interaction and document understanding.

## 🔧 Features

- **LLM-Powered Chat**  
  Runs completely offline using local models (e.g., `mistral:7b`) via [Ollama](https://ollama.com).

- **Streamlit Interface**  
  A modern web chat UI with avatars, personality configuration, and persistent chat memory.

- **📄 Document Q&A**  
  Upload `.pdf` or `.txt` files and ask questions based on their content using a FAISS-powered retriever.

- **💾 Persistent Chat History**  
  Saves chat sessions to disk and reloads them automatically.

- **🎙️ Voice Input & Output**  
  Use your microphone for speech-to-text and hear the bot reply via text-to-speech. Toggle from the sidebar.

- **🧠 Configurable Personality**  
  Customise the assistant’s tone and behavior via a system prompt field in the sidebar.

---

## 🛠 Tech Stack

- Python
  - ![LangChain](https://img.shields.io/badge/LangChain-000000?style=for-the-badge&logo=chainlink&logoColor=white)
  - ![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=openai&logoColor=white)
  - ![FAISS](https://img.shields.io/badge/FAISS-0099CC?style=for-the-badge&logo=vector&logoColor=white)
  - ![Ollama Embeddings](https://img.shields.io/badge/Ollama_Embeddings-412991?style=for-the-badge&logo=openai&logoColor=white)
  - ![PyPDFLoader](https://img.shields.io/badge/PyPDFLoader-4B8BBE?style=for-the-badge&logo=readthedocs&logoColor=white)
  - ![TextLoader](https://img.shields.io/badge/TextLoader-888888?style=for-the-badge&logo=readthedocs&logoColor=white)
  - ![SpeechRecognition](https://img.shields.io/badge/SpeechRecognition-007ACC?style=for-the-badge&logo=microphone&logoColor=white)
  - ![pyttsx3](https://img.shields.io/badge/pyttsx3-B5651D?style=for-the-badge&logo=soundcloud&logoColor=white)

---

## 🚀 Getting Started

```bash
# Clone and install requirements
pip install -r requirements.txt

# Run the chatbot
streamlit run UIChat.py
