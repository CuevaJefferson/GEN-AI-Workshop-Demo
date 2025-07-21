import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import glob


# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


def load_all_pdfs(directory="documents/"):
    all_docs = []
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    for file in pdf_files:
        loader = PyMuPDFLoader(file)
        all_docs.extend(loader.load())
    return all_docs


# Load and split PDF
def load_documents():
    docs = load_all_pdfs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)


# Create FAISS vector store
def create_vector_store(docs):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(docs, embeddings)


# Set up Groq LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile", api_key=groq_api_key, temperature=0.7
)


def main():
    print("Loading and indexing your PDFs...")
    docs = load_documents()
    vectorstore = create_vector_store(docs)
    retriever = vectorstore.as_retriever()

    # Set up RAG chain
    qa = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=True
    )

    print("Ask anything about the document.\n")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Bye!")
            break

        result = qa.invoke({"query": query})
        print("\nAnswer:", result["result"])
        print("------")
        print("Sources:")
        for doc in result["source_documents"]:
            print(f"- Page from {doc.metadata.get('source', 'unknown')}")
            # print(f"    Page Content {doc.page_content}")


if __name__ == "__main__":
    main()
