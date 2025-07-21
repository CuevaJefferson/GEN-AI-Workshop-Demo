import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough

# Load Groq API key from .env
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ------------------------------
# 🔹 LangChain Component: PromptTemplate
# ------------------------------
template = PromptTemplate(
    input_variables=["topic"], template="Explain {topic} like I'm five."
)

# ------------------------------
# 🔹 LangChain Component: ChatGroq (LLM)
# ------------------------------
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=groq_api_key,
)

# ------------------------------
# 🔹 LangChain Component: Runnable Pipeline
# ------------------------------
chain = {"topic": RunnablePassthrough()} | template | llm


# ------------------------------
# Main chatbot loop
# ------------------------------
def main():
    print("Explain Like I’m 5 (type 'exit' to quit)\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        response = chain.invoke(user_input)
        print("Bot:", response.content)


if __name__ == "__main__":
    main()
