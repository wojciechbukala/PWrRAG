from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

# --- VECTORSTORE ---

CHROMA_PATH = "./chroma_db"
COLLECTION = "PWrRAG"

EMBEDDING_MODEL = "embeddinggemma"
LLM_MODEL = "gemma3:4b"

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

def reset_vectorstore():
    db = Chroma(
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION,
        embedding_function=embeddings
    )
    db.delete_collection()

def set_vectorstore(chunks: list):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION
    )
    return vectorstore

def add_to_vectorstore(chunks: list):
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    return vectorstore

def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION,
        embedding_function=embeddings
    )

def verify_vectorstore():
    try:
        vectorstore = get_vectorstore()
        # Check number of documents
        doc_count = vectorstore._collection.count()
        print(f"Database consists of {doc_count} documents.")
        
        if doc_count == 0:
            print("[WARNING]: Database is empty")
            return

    except Exception as e:
        print(f"Error veryfing chroma DB: {str(e)}")


# --- LLM ---

def get_llm(model: str = LLM_MODEL):
    return ChatOllama(
        model=model,
        temperature=0.1,
        num_predict=800,
    )

