import requests
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

# Define the URL of your FastAPI endpoint
url = "http://localhost:8000/query"

# Initialize the session state
st.session_state.setdefault("chat_history", [])

# Function to handle the new chat button click
def new_chat():
    st.session_state.chat_history = []  # Clear the chat history

# Streamlit app
def app():
    st.title("Doctor's Medical Assistant")
    st.sidebar.button("New Chat", on_click=new_chat)
    st.image("2.jpg", width=300)

    # Display Welcome message
    st.write("<span style='font-size:20px; font-weight:bold;'>Welcome! How Can I Help You</span>", 
             unsafe_allow_html=True)

    # Placeholder text for the input box
    input_placeholder = st.empty()
    input_text = input_placeholder.text_input("", key="user_input", help="Type your question here...")

    # JavaScript to handle the placeholder behavior
    placeholder_script = f"""
    <script>
        const inputElement = document.querySelector('input[data-baseweb="input"]');
        inputElement.placeholder = "Enter your question";
    </script>
    """
    st.markdown(placeholder_script, unsafe_allow_html=True)

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
