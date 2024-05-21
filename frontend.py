import requests
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

url = "https://shahabkahn-medical-assistant.hf.space/query"

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "input_reset" not in st.session_state:
    st.session_state.input_reset = False

def new_chat():
    st.session_state.chat_history = []
    st.session_state.input_reset = True

def app():
    st.title("Doctor's Medical Assistant")
    st.sidebar.button("New Chat", on_click=new_chat)
    st.image("2.jpg", width=300)

    st.write("<span style='font-size:20px; font-weight:bold;'>Welcome! How Can I Help You</span>", 
             unsafe_allow_html=True)

    if st.session_state.input_reset:
        input_text = st.text_input("", placeholder="Enter your question here...", 
                                   key="user_input_reset", help="Type your question here...")
        if input_text:
            st.session_state.user_input = input_text
            st.session_state.input_reset = False
            st.experimental_rerun()
    else:
        input_text = st.text_input("", placeholder="Enter your question here...", 
                                   key="user_input", help="Type your question here...")

    submit_button = st.button("➡️")
    if submit_button:
        user_input = st.session_state.user_input.strip()
        if user_input:
            payload = {"question": user_input}
            try:
                response = requests.post(url, json=payload)
                if response.ok:
                    answer = response.json().get("answer")
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.session_state.input_reset = True
                    st.experimental_rerun()
                else:
                    st.error(f"Error: {response.status_code} {response.text}")
            except requests.RequestException as e:
                st.error(f"Error: {e}")

    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.write(f"**You:** {chat['content']}")
        else:
            st.write(f"**Assistant:** {chat['content']}")

if __name__ == "__main__":
    app()
