# PWrRAG

## Opis projektu

Tematem projektu jest stworzenie asystenta dla pracowników i studentów Politechniki Wrocławskiej do szybkiego i sprawnego wyszukiwania informacji zawartych w dokumentach formalnych obowiązujących na uczelni. Przez dokumnety formalne rozumie się regulaminy studiów, wszelkiego rodzaju podania, statuty, wzory dokumnetów itp. 

W tym celu wykorzystana zostanie technika RAG (Retrieval-Augmented Generation), umożliwiająca modelom generatywnym sztucznej inteligencji wyszukiwanie i włączanie do okna kontekstowego, zewnętrznych informacji. W ten sposób można modyfikować zachowanie dużego modelu językowego bez konieczności kosztownego fine-tuningu.

W projekcie wykorzystane zostaną takie technologie jak: LangChain, LLM z ogólnodostępnej bazy Hugging Face, ChromaDB/FAISS jako baza wektorowa.

## Plan pracy
1. Zdefiniowanie problemów, struktury projektu i podział pracy.
2. Pobranie danych formalnych PWr z dostępnych źródeł lub poprzez scrapowanie.
3. Filtracja i przetworzenie danych, w celu pozbycia się zbędnych, nieaktualnych lub niepoprawnych dokumentów.
4. Osadzenie danych w bazie wektorowej (tokenizacja, indeksowanie).
5. Budowa pipeline'u generacji odpowiedzi.
6. Testowanie i dostosowanie pipeline'u (porównanie modeli oraz technik prompt engineeringu).
7. Ubogacenie generowanych odpowiedzi o elementy dodatkowe (np. odnośnik do cytowanego dokumentu).

## Harmonogram
- 2026-04-09 - Stworzony zbiór dokumnetów formalnych PWr.
- 2026-04-23 - Dokumenty przetworzone do formy bazy wektorowej.
- 2026-05-07 - Budowa pipeline'u.
- 2026-05-19 - Działająca generacja z wykorzystaniem minimum jednego modelu testowego.
- 2026-06-18 - Finalna forma projektu i raport z eksperymentów.

## Bibliografia
Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., & Wang, H. (2024). Retrieval-Augmented Generation for Large Language Models: A Survey. arXiv preprint arXiv:2312.10997.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. Advances in Neural Information Processing Systems (NeurIPS), 33, 9459–9474.

[Hugging Face LLM Course](https://huggingface.co/learn/llm-course/)

[LangChain RAG pipeline example](https://github.com/langchain-ai/rag-from-scratch)

[OpenAI API docs](https://developers.openai.com/api/docs)


## Sprawdzanie bazy wektorowej chromaDB

Aby sprawdzić lokalną bazę danych skopiuj w terminalu ten fragment:
```
cd ~/PWrRAG/PWrRAG && source venv/bin/activate && python - <<'EOF'
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

# Lista kolekcji
for col in client.list_collections():
    print(f"Kolekcja: {col.name}, liczba dokumentów: {col.count()}")

# Szczegóły kolekcji PWrRAG
col = client.get_collection("PWrRAG")
print(f"\nLiczba chunków: {col.count()}")

# Przykładowe 3 chunki
result = col.peek(limit=3)
for i, (doc, meta) in enumerate(zip(result['documents'], result['metadatas'])):
    print(f"\n--- Chunk {i+1} ---")
    print(f"Metadata: {meta}")
    print(f"Tekst: {doc[:300]}...")
EOF

```

