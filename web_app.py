import os
import openai
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

# Function to initialize the LLM client (OpenAI or OpenRouter)
def create_llm():
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Error: The OPENAI_API_KEY or OPENROUTER_API_KEY environment variable is not set.")
        st.stop()
    # If using OpenRouter, configure base and key
    if os.getenv("OPENROUTER_API_KEY"):
        openai.api_key = api_key
        openai.api_base = "https://openrouter.ai/api/v1"
        try:
            models = openai.Model.list()
            available = [m.get("id") for m in models.get("data", [])]
            model_name = next((m for m in available if m), "gpt-3.5-turbo")
        except Exception:
            model_name = "gpt-3.5-turbo"
        return ChatOpenAI(
            temperature=0.7,
            model_name=model_name,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
        )
    # Fallback to OpenAI
    return ChatOpenAI(temperature=0.7, openai_api_key=api_key)

# Initialize session state
if "conversation" not in st.session_state:
    llm = create_llm()
    st.session_state.conversation = ConversationChain(llm=llm, verbose=False)

st.title("LangChain Chatbot")

# Display chat history
if "history" not in st.session_state:
    st.session_state.history = []

for entry in st.session_state.history:
    speaker, msg = entry
    if speaker == "user":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Bot:** {msg}")

# User input
user_input = st.text_input("You:", key="input")
if st.button("Send") and user_input:
    st.session_state.history.append(("user", user_input))
    with st.spinner("Waiting for response..."):
        try:
            response = st.session_state.conversation.predict(input=user_input)
        except Exception as e:
            if getattr(e, 'code', None) == 'insufficient_quota':
                response = "⚠️ You've hit quota limit. Please top up your account."
            else:
                response = f"Error: {e}"
    st.session_state.history.append(("bot", response))
    st.experimental_rerun() 