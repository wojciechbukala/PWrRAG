import os
from dotenv import load_dotenv

load_dotenv()
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['HUGGINGFACEHUB_API_KEY'] = os.getenv('HUGGINGFACE_API_KEY')

from Models import get_vectorstore



### QUERY TRANSLATION ###

### QUERY CONSTRUCTION ###

### RETRIEVAL ##

def retrieval():
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5,
        },
    )
    return retriever

## GENERATION ##