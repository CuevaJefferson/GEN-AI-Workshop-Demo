import os
from dotenv import load_dotenv
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_groq import ChatGroq

# Load API key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Set up LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    api_key=groq_api_key,
    temperature=0.7,
)

# Define prompt with message history
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Create chain
chain = prompt | llm

# Wrap chain with memory
memory_chain = RunnableWithMessageHistory(
    chain,
    lambda session_id: InMemoryChatMessageHistory(),  # per session
    input_messages_key="messages",
)


# CLI chat loop
def main():
    print("Chatbot with memory (RunnableWithMessageHistory)")
    session_id = "user-session-1"  # can be dynamic per user
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        result = memory_chain.invoke(
            {"messages": chat_history + [{"role": "user", "content": user_input}]},
            config={"configurable": {"session_id": session_id}},
        )

        response = result.content
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response})

        print("Bot:", response)


if __name__ == "__main__":
    main()
