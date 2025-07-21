import os
from dotenv import load_dotenv
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS


# ------------------------------
# Load API Key
# ------------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


# ------------------------------
# Calculator Tool
# ------------------------------
def calculator_tool_func(query: str) -> str:
    try:
        return str(eval(query))
    except Exception as e:
        return f"Error: {e}"


calculator_tool = Tool(
    name="Calculator",
    func=calculator_tool_func,
    description="Useful for solving math expressions. Input should be a valid Python-style expression like '12 / (3 + 1)'.",
)


# ------------------------------
# Fake Database Tool
# ------------------------------
def fake_database_tool_func(query: str) -> str:
    fake_db = {
        "john": "John is a software engineer at OpenAI.",
        "sara": "Sara is a data scientist in the health industry.",
        "project alpha": "Project Alpha is a confidential AI research initiative.",
    }

    key = query.lower().strip()
    return fake_db.get(key, f"No record found for '{query}' in the fake database.")


database_tool = Tool(
    name="FakeDatabase",
    func=fake_database_tool_func,
    description="Looks up simulated user or project info from a fake database. Try inputs like 'John', 'Sara', or 'Project Alpha'.",
)


# ------------------------------
# DuckDuckGo Internet Search Tool
# ------------------------------
def duckduckgo_search_tool_func(query: str) -> str:
    results = DDGS().text(query, max_results=3)
    summaries = [f"{r['title']}: {r['href']}" for r in results]
    return "\n".join(summaries) if summaries else "No relevant results found."


search_tool = Tool(
    name="InternetSearch",
    func=duckduckgo_search_tool_func,
    description="Searches the web for general knowledge and current information.",
)

# ------------------------------
# Combine Tools
# ------------------------------
# tools = [calculator_tool, database_tool]

tools = [calculator_tool, database_tool, search_tool]


# ------------------------------
# Groq LLM + Agent
# ------------------------------
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile", temperature=0.7, api_key=groq_api_key
)

agent = initialize_agent(
    tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)


# ------------------------------
# Main Loop
# ------------------------------
def main():
    print("Tool-Using AI Agent")
    print("Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        response = agent.invoke(user_input)
        print("Bot:", response)


if __name__ == "__main__":
    main()
