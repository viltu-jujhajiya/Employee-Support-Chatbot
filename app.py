import streamlit as st
from src.main import escb

# This must be the FIRST Streamlit command
st.set_page_config(page_title="Employee Support Chatbot", layout="centered")

@st.cache_resource
def load_chatbot():
    return escb()

chatbot = load_chatbot()

st.title("💼 Employee Support Chatbot")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history (scrolls up as you chat)
for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)

# Input box at the bottom
user_input = st.chat_input("Ask a question")

if user_input:
    # Display user message
    st.chat_message("user").markdown(user_input)

    with st.spinner("Thinking..."):
        response = chatbot.main(user_input)

    # Display bot response
    st.chat_message("assistant").markdown(response)

    # Save to history
    st.session_state.chat_history.append((user_input, response))