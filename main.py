import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
model = ChatGroq(model="llama-3.3-70b-versatile", max_retries=1)

vector_store = Chroma(
    collection_name="rag_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True,
)

system_prompt = """Answer the question using the context below.
If multiple figures appear, pick the one that directly answers the question.
Only say "I don't know" if the answer is genuinely not in the context.

Context:
{context}
"""

def build_vectorstore(file_path):


    existing = vector_store.get()
    if existing["ids"]:
        vector_store.delete(ids=existing["ids"])

        
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    chunks = text_splitter.split_documents(docs)
    vector_store.add_documents(chunks)

def answer_question(query):
    retrieved_docs = vector_store.similarity_search(query, k=6)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    messages = [
        {"role": "system", "content": system_prompt.format(context=context)},
        {"role": "user", "content": query},
    ]
    return model.invoke(messages).content

if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or type 'quit'): ")
        if query.lower() == "quit":
            break
        print(answer_question(query))