from langchain_core.messages import SystemMessage, HumanMessage

from Models import get_vectorstore, get_llm


### QUERY TRANSLATION ###

### QUERY CONSTRUCTION ###

### RETRIEVAL ##

def retrieval(k: int = 4):
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    return retriever

## GENERATION ##

def format_retrieved_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

SYSTEM_PROMPT = """Jesteś asystentem na Politechnice Wrocławskiej.
Odpowiedz na zadane pytanie prioretyzując podany kontekst.
Twoim domyślnym językiem jest polski w tonie oficjalnej, trzymaj się tej zasady, chyba że jasno zostaniesz poproszony o zmianę.
Jeśli nie jesteś pewny odpowiedzi, odpowiedz: 'Nie znam odpowiedzi na to pytanie.'"""

def generation(question: str, model: str = None, k: int = 4) -> tuple[str, list]:
    retriever = retrieval(k=k)
    retrieved = retriever.invoke(question)
    context = format_retrieved_docs(retrieved)

    llm = get_llm(model=model) if model else get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Kontekst:\n{context}\n\nPytanie: {question}"),
    ]
    response = llm.invoke(messages)
    return response.content, retrieved


### Tests
if __name__ == '__main__':
    question = "Jakie są prawa studenta na politechnice wrocławskiej"

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