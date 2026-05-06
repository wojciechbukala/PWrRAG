import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from openai import OpenAI

from Models import get_vectorstore


### QUERY TRANSLATION ###

### QUERY CONSTRUCTION ###

### RETRIEVAL ##

def retrieval():
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 20,
        },
    )
    return retriever

## GENERATION ##

def format_retrieved_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generation(question: str) -> str:
    retriever = retrieval()

    retrieved = retriever.invoke(question)

    context = format_retrieved_docs(retrieved)

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ["HUGGINGFACE_API_KEY"],
    )

    response = client.responses.create(
        model="CYFRAGOVPL/PLLuM-12B-chat:featherless-ai",
        instructions="""Jesteś asystentem na Politechnice Wrocławskiej.
Odpowiedz na zadane pytanie prioretyzując podany kontekst.
Twoim domyślnym językiem jest polski w tonie ofcjalnej, trzymaj się tej zasady, chyba że jasno zostaniesz poproszony o zmianę.
Jeśli nie jesteś pewny odpowiedzi, odpowiedz: 'Nie znam odpowiedzi na to pytanie.'""",
        input=[
            {
                "role": "developer",
                "content": f"""Kontekst: {context}"""
            },
            {
                "role": "user",
                "content": f"""Pytanie: {question}"""
            }
        ],
        max_output_tokens=30000
    )
    return response.output_text


### Tests
if __name__ == '__main__':
    question = "Jakie są zasady rekrutacji na Inforamtyczne Systemy Automatyki - studia II stopnia."

    # Test retrieval
    retriever = retrieval()
    retrieved = retriever.invoke(question)
    print("=== Pobrane fragmenty ===")
    for i, doc in enumerate(retrieved):
        print(f"Fragment {i+1}: {doc.page_content}\n")

    # Test pełnego pipeline RAG (retrieval + generation)
    print("=== Odpowiedź modelu ===")
    answer = generation(question)
    print(answer)