import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

# Load API key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile", temperature=0.7, api_key=groq_api_key
)

# Simple prompt template (no history)
prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer: {question}"
)


def main():
    print("🤖 Chatbot with NO memory. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Bot: Goodbye!")
            break
        chain_input = prompt.invoke({"question": user_input})
        response = llm.invoke(chain_input)
        print("Bot:", response.content)


if __name__ == "__main__":
    main()
