import os
import openai
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

def main():
    # Get API key from OpenAI or OpenRouter
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not openai_key:
        print("Error: The OPENAI_API_KEY or OPENROUTER_API_KEY environment variable is not set.")
        return

    # Determine which key is set
    router_key = os.getenv("OPENROUTER_API_KEY")
    if router_key:
        # Configure OpenAI client for OpenRouter
        openai.api_key = router_key
        openai.api_base = "https://openrouter.ai/api/v1"
        # Try to list models and pick the first one
        try:
            models = openai.Model.list()
            available = [m.get("id") for m in models.get("data", [])]
            model_name = next((m for m in available if m), "gpt-3.5-turbo")
            print(f"Using OpenRouter model: {model_name}")
        except Exception as e:
            print(f"Warning: could not fetch OpenRouter models ({e}), defaulting to gpt-3.5-turbo")
            model_name = "gpt-3.5-turbo"
        llm = ChatOpenAI(
            temperature=0.7,
            model_name=model_name,
            openai_api_key=router_key,
            openai_api_base="https://openrouter.ai/api/v1",
        )
    else:
        # Use standard OpenAI key
        llm = ChatOpenAI(temperature=0.7, openai_api_key=openai_key)

    conversation = ConversationChain(llm=llm, verbose=False)
    print("Chatbot: Hello! I'm a simple LangChain chat bot. Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break
        try:
            response = conversation.predict(input=user_input)
        except Exception as e:
            # Handle OpenAI quota errors gracefully
            if getattr(e, 'code', None) == 'insufficient_quota':
                print("Chatbot: ⚠️ You've hit OpenAI's quota limit. Please top up your account and try again.")
                continue
            raise
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main() 