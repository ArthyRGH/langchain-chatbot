import os
import openai
import tkinter as tk
from tkinter import scrolledtext
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

def create_llm():
    # Get API key from OpenAI or OpenRouter
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not openai_key:
        raise ValueError("Error: The OPENAI_API_KEY or OPENROUTER_API_KEY environment variable is not set.")
    router_key = os.getenv("OPENROUTER_API_KEY")
    if router_key:
        # Configure OpenRouter
        openai.api_key = router_key
        openai.api_base = "https://openrouter.ai/api/v1"
        # Select a working model
        try:
            models = openai.Model.list()
            available = [m.get("id") for m in models.get("data", [])]
            model_name = next((m for m in available if m), "gpt-3.5-turbo")
        except Exception:
            model_name = "gpt-3.5-turbo"
        return ChatOpenAI(
            temperature=0.7,
            model_name=model_name,
            openai_api_key=router_key,
            openai_api_base="https://openrouter.ai/api/v1",
        )
    # Fallback to OpenAI
    return ChatOpenAI(temperature=0.7, openai_api_key=openai_key)

def main():
    llm = create_llm()
    conversation = ConversationChain(llm=llm, verbose=False)

    root = tk.Tk()
    root.title("LangChain Chatbot")

    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, width=80, height=20)
    text_area.pack(padx=10, pady=10)

    entry_frame = tk.Frame(root)
    entry_frame.pack(fill=tk.X, padx=10, pady=(0,10))
    entry = tk.Entry(entry_frame)
    entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
    entry.focus()
    send_button = tk.Button(entry_frame, text="Send")
    send_button.pack(side=tk.RIGHT)

    def append_text(msg):
        text_area.configure(state=tk.NORMAL)
        text_area.insert(tk.END, msg)
        text_area.configure(state=tk.DISABLED)
        text_area.yview(tk.END)

    def send_message(event=None):
        user_text = entry.get().strip()
        if not user_text:
            return
        entry.delete(0, tk.END)
        append_text(f"User: {user_text}\n")
        if user_text.lower() in ["exit", "quit"]:
            append_text("Chatbot: Goodbye!\n")
            root.quit()
            return
        try:
            response_text = conversation.predict(input=user_text)
        except Exception as e:
            if getattr(e, 'code', None) == 'insufficient_quota':
                append_text("Chatbot: ⚠️ You've hit OpenAI's quota limit. Please top up your account and try again.\n")
            else:
                append_text(f"Error: {e}\n")
            return
        append_text(f"Chatbot: {response_text}\n")

    entry.bind("<Return>", send_message)
    send_button.configure(command=send_message)

    append_text("Chatbot: Hello! I'm a simple LangChain chat bot. Type 'exit' or 'quit' to stop.\n")
    root.mainloop()

if __name__ == "__main__":
    main() 