import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

# Session state init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()
    st.session_state.messages = []

# LLM setup
llm = ChatOllama(model="mistral:7b")

# Page setup
st.set_page_config(page_title="Local Chatbot", page_icon="")
st.title("Daniel's ChatBot")

# Display chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)

# Input box
if user_input := st.chat_input("Ask me anything..."):
    # Display user input
    st.chat_message("user").write(user_input)
    st.session_state.chat_history.add_user_message(user_input)

    # Generate reply
    response = llm.invoke(st.session_state.chat_history.messages + [HumanMessage(content=user_input)])
    st.session_state.chat_history.add_ai_message(response.content)

    # Store message pairs for UI replay
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.session_state.messages.append(AIMessage(content=response.content))

    # Display assistant reply
    st.chat_message("assistant").write(response.content)
