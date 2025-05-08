
import streamlit as st

st.set_page_config(page_title="Home", page_icon="🏠")
st.title("Welcome!")

st.write("Navigate to:")
st.page_link("pages/1_Chatbot.py", label="🤖 Chatbot")
st.page_link("pages/2_Graph_Generator.py", label="📊 Graph Generator")
