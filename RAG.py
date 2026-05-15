import os
from dotenv import load_dotenv

load_dotenv()

# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

from Models import get_vectorstore


### QUERY TRANSLATION ###

### QUERY CONSTRUCTION ###

### RETRIEVAL ##

def retrieval():
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,  # Ile finalnie fragmentów dajemy do LLMa
            "fetch_k": 30,  # Z ilu dokumentów ma wybierać te najbardziej różnorodne
            "lambda_mult": 0.5  # Balans między podobieństwem a różnorodnością (0.5 to dobry start)
        },
    )
    return retriever
    # vectorstore = get_vectorstore()
    # retriever = vectorstore.as_retriever(
    #     search_type="similarity",
    #     search_kwargs={
    #         "k": 20,
    #     },
    # )
    # return retriever

## GENERATION ##

def format_retrieved_docs(docs):
    formatted_chunks = []
    for i, document in enumerate(docs):
        # Pobranie źródła z metadanych
        source = document.metadata.get("source", "Brak zródła")

        # Tworzenie bloku dla modelu
        chunk_text = f"--- Fragment {i + 1} ---\nŹródło: {source}\nTreść: {document.page_content}"
        formatted_chunks.append(chunk_text)

    # return "\n\n".join(document.page_content for document in docs)
    return "\n\n".join(formatted_chunks)

def generation(question: str) -> str:
    retriever = retrieval()

    retrieved = retriever.invoke(question)

    context = format_retrieved_docs(retrieved)

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ["HUGGINGFACE_API_KEY"],
    )

    messages_list = [
        ChatCompletionSystemMessageParam(
            role="system",
            content="""Jesteś precyzyjnym asystentem na Politechnice Wrocławskiej.
                    Odpowiadasz WYŁĄCZNIE na podstawie dostarczonego KONTEKSTU.

BEZWZGLĘDNE ZASADY:
1. NIE WOLNO Ci korzystać z wiedzy zewnętrznej ani wymyślać odpowiedzi.
2. Jeśli KONTEKST nie dotyczy wprost pytania, MUSISZ odpowiedzieć: "Przepraszam, ale nie znalazłem odpowiedzi na to pytanie w moich dokumentach."
3. Jeśli udzielisz odpowiedzi na podstawie dokumentów, ZAWSZE podaj źródła pod odpowiedzią, na przykład:
[Źródła]:
- [URL 1]
- [URL 2]"""
        ),
        ChatCompletionUserMessageParam(
            role="user",
            content=f"Kontekst:\n{context}\n\nPytanie: {question}"
        )
    ]

    response = client.chat.completions.create(
        model="CYFRAGOVPL/PLLuM-12B-chat:featherless-ai",
        messages=messages_list,
        max_tokens=4096
    )

    return response.choices[0].message.content

#     response = client.responses.create(
#         model="CYFRAGOVPL/PLLuM-12B-chat:featherless-ai",
#         instructions="""Jesteś asystentem na Politechnice Wrocławskiej.
# Odpowiedz na zadane pytanie prioretyzując podany kontekst.
# Twoim domyślnym językiem jest polski w tonie ofcjalnej, trzymaj się tej zasady, chyba że jasno zostaniesz poproszony o zmianę.
# Jeśli nie jesteś pewny odpowiedzi, odpowiedz: 'Nie znam odpowiedzi na to pytanie.
# Na końcu swojej odpowiedzi zawsze podaj listę źródeł, z których korzystałeś, używając linków podanych we fragmentach kontekstu.'""",
#         input=[
#             {
#                 "role": "developer",
#                 "content": f"""Kontekst: {context}"""
#             },
#             {
#                 "role": "user",
#                 "content": f"""Pytanie: {question}"""
#             }
#         ],
#         max_output_tokens=30000
#     )
#     return response.output_text


### Tests
if __name__ == '__main__':
    question = "Jakie są zasady rekrutacji na Automatykę i Robotykę - studia II stopnia."

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