import os
import openai
import streamlit as st
import requests
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

st.set_page_config(page_title="LangChain Chatbot", page_icon="🤖")

st.sidebar.header("Configuration")
api_source = st.sidebar.radio("API Source", ["OpenAI", "OpenRouter"])

# API key and model selection
api_key = ""
model_opts = []
model_name = ""

if api_source == "OpenRouter":
    api_key = st.sidebar.text_input("OpenRouter API Key", type="password", value=os.getenv("OPENROUTER_API_KEY", ""))
    if api_key:
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            resp = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=10)
            resp.raise_for_status()
            models = resp.json().get("data") or []
            model_opts = [m["id"] for m in models if m.get("id")]
            if not model_opts:
                model_opts = ["openrouter/openai/gpt-3.5-turbo"]
        except Exception as e:
            st.sidebar.warning(f"Could not fetch models from OpenRouter: {e}. Using default.")
            model_opts = ["openrouter/openai/gpt-3.5-turbo"]
    else:
        model_opts = ["openrouter/openai/gpt-3.5-turbo"]
    model_name = st.sidebar.selectbox("Model Name", options=model_opts, index=0)
else:
    api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    model_opts = ["gpt-3.5-turbo", "gpt-4"]
    model_name = st.sidebar.selectbox("Model Name", options=model_opts, index=0)

# Button to start chat
if st.sidebar.button("Start Chat"):
    st.session_state['chat_ready'] = True
    st.session_state['api_source'] = api_source
    st.session_state['api_key'] = api_key
    st.session_state['model_name'] = model_name
    st.session_state['history'] = []
    st.session_state['conversation'] = None

if st.sidebar.button("Reset Conversation"):
    st.session_state.clear()

st.title("🤖 LangChain Chatbot")

# Only show chat UI if chat is ready
if st.session_state.get('chat_ready'):
    if st.session_state.get('conversation') is None:
        llm_kwargs = {
            "temperature": 0.7,
            "model_name": st.session_state['model_name'],
            "openai_api_key": st.session_state['api_key'],
            "request_timeout": 30,
            "max_tokens": 512,
        }
        if st.session_state['api_source'] == "OpenRouter":
            llm_kwargs["openai_api_base"] = "https://openrouter.ai/api/v1"
        llm = ChatOpenAI(**llm_kwargs)
        st.session_state.conversation = ConversationChain(llm=llm, verbose=False)
        st.session_state.history = []

    # Chat input with immediate response
    user_input = st.chat_input("Type your message here...", key="chat_input")
    if user_input and (not st.session_state.history or st.session_state.history[-1][1] != user_input):
        st.session_state.history.append(("user", user_input))
        with st.spinner("Generating response..."):
            import time
            start = time.time()
            try:
                response = st.session_state.conversation.predict(input=user_input)
            except Exception as e:
                error_str = str(e)
                if 'Input validation error' in error_str:
                    response = "⚠️ Input validation error: Please try a different model or reduce your prompt size/max_tokens."
                elif 'credits' in error_str or 'max_tokens' in error_str:
                    response = "⚠️ Not enough credits or too many tokens requested. Please reduce max_tokens or top up your account."
                elif 'auth credentials' in error_str or '401' in error_str:
                    response = "⚠️ Authentication error: Please check your API key and provider selection."
                else:
                    response = f"Error: {e}"
            end = time.time()
            st.session_state.history.append(("assistant", response))
            st.info(f"Response time: {end - start:.2f} seconds")

    # Display chat history
    for speaker, msg in st.session_state.history:
        if speaker == "user":
            st.chat_message("user").write(msg)
        else:
            st.chat_message("assistant").write(msg)
else:
    st.info("Please enter your API key, select a model, and click 'Start Chat' to begin.") 