from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_chroma import Chroma
import torch
import os
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HUGGINGFACE_API_KEY")
if hf_token:
    os.environ["HUGGINGFACE_TOKEN"] = hf_token
else:
    print("⚠️  HUGGINGFACE_API_KEY not set in .env")

# --- VECTORSTORE ---

CHROMA_PATH = "./chroma_db"
COLLECTION = "PWrRAG"

ENCODER_MODEL_NAME = "intfloat/multilingual-e5-small"

MODEL_KWARGS = {'device': 'cpu'}
ENCODE_KWARGS = {'normalize_embeddings': True}

embeddings = HuggingFaceEmbeddings(
    model_name=ENCODER_MODEL_NAME,
    model_kwargs=MODEL_KWARGS,
    encode_kwargs=ENCODE_KWARGS
)

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
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

def get_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True
    )

    return HuggingFacePipeline(pipeline=pipe)

