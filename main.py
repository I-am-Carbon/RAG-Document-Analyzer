import os
import uuid
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import cohere

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
co = cohere.Client(os.getenv("COHERE_API_KEY"))




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
sessions = {}
def build_vectorstore(file_path, session_id = None):

    if session_id == None:
       session_id = str(uuid.uuid4())

    session_vector_store = Chroma(
        collection_name=f"rag_collection_{session_id}",
        embedding_function=embeddings,
        persist_directory=f"./chroma_db_{session_id}",
    )

        
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    chunks = text_splitter.split_documents(docs)
    session_vector_store.add_documents(chunks)

    # inside build_vectorstore, after creating chunks:
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 20

    vector_retriever = session_vector_store.as_retriever(search_kwargs={"k": 20})

    session_ensemble =  EnsembleRetriever(
       retrievers=[bm25_retriever, vector_retriever],
       weights=[0.5, 0.5]
    )

    sessions[session_id] = {
        "vector_store": session_vector_store,
        "ensemble": session_ensemble,
    }

    return session_id

def answer_question(query ,session_id):
    session = sessions.get(session_id)
    if session is None:
        return "No document uploaded for this session. Please upload a document first."

    ensemble = session["ensemble"]
    retrieved_docs = ensemble.invoke(query)
    
    results = co.rerank(
       model="rerank-v4.0-fast",
       query = query,
       documents = [doc.page_content for doc in retrieved_docs],
       top_n=4
    )

    top_docs = []
    for r in results.results:
      top_docs.append(retrieved_docs[r.index])
    
    context = ""
    for doc in top_docs:
      context += doc.page_content + "\n\n"

    messages = [
        {"role": "system", "content": system_prompt.format(context=context)},
        {"role": "user", "content": query},
    ]
    return model.invoke(messages).content

if __name__ == "__main__":
    session_id = build_vectorstore(input("Enter PDF path: "))
    while True:
        query = input("\nAsk a question (or type 'quit'): ")
        if query.lower() == "quit":
            break
        print(answer_question(query, session_id))