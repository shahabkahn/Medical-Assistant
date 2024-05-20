import requests
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

# Define the URL of your FastAPI endpoint
url = "https://shahabkahn-medical-assistant.hf.space/query"

# Initialize the session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Function to handle the new chat button click
def new_chat():
    st.session_state.chat_history = []  # Clear the chat history
    st.session_state.user_input = ""    # Clear the user input

# Streamlit app
def app():
    st.title("Doctor's Medical Assistant")
    st.sidebar.button("New Chat", on_click=new_chat)
    st.image("2.jpg", width=300)

    # Display Welcome message
    st.write("<span style='font-size:20px; font-weight:bold;'>Welcome! How Can I Help You</span>", 
             unsafe_allow_html=True)

    # Placeholder text for the input box
    input_text = st.text_input("Let's Chat", placeholder="Enter your question here...", 
                               key="user_input", help="Type your question here...")

    # Handle form submission
    submit_button = st.button("➡️")
    if submit_button:
        user_input = input_text.strip()
        if user_input:
            # Create the request payload
            payload = {"question": user_input}
            try:
                # Send the POST request to the FastAPI endpoint
                response = requests.post(url, json=payload)
                # Check if the request was successful
                if response.ok:
                    # Get the answer from the FastAPI endpoint
                    answer = response.json().get("answer")
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"Error: {response.status_code} {response.text}")
            except requests.RequestException as e:
                st.error(f"Error: {e}")

    # Display chat history
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.write(f"**You:** {chat['content']}")
        else:
            st.write(f"**Assistant:** {chat['content']}")

if __name__ == "__main__":
    app()
