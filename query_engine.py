import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Load vector store
def load_vector_store(store_path="vector_store"):
    embeddings = OpenAIEmbeddings(api_key=api_key)
    return FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)

# Create QA engine
def create_qa_engine(store_path="vector_store"):
    db = load_vector_store(store_path)
    retriever = db.as_retriever()
    llm = OpenAI(api_key=api_key, temperature=0.1)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa

# Ask a question
def ask_question(question, course_id):
    store_path = os.path.join("vectorstores", course_id)  # ✅ Correct path
    qa = create_qa_engine(store_path)
    return qa.run(question)


# For testing:
if __name__ == "__main__":
    q = input("❓ Ask a question: ")
    print("💬 Answer:", ask_question(q))
